import rclpy
import math
from rclpy.node import Node
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup

from sensor_msgs.msg import NavSatFix, PointCloud2
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import PoseStamped, TwistStamped, Twist
from std_msgs.msg import Empty, Header
import sensor_msgs_py.point_cloud2 as pc2

from nav2_msgs.action import FollowPath

from goof_an_odd_husky_common.config import (
    TOPICS, USE_GPS, DEBUG, HEARTBEAT_TIMEOUT_SEC,
    OSM_RELATION_ID, MAX_PATH_EDGE, MAX_TRAJECTORY_DISTANCE,
    SAFETY_RADIUS, BARRIER_OFFSET
)
from goof_an_odd_husky_common.qos import LATCHED_QOS
from goof_an_odd_husky_common.types import GpsCoord

from goof_an_odd_husky.global_navigation.path_manager import GlobalPathManager
from goof_an_odd_husky.local_navigation.local_goal_selector import LocalGoalSelector
from goof_an_odd_husky_common.helpers import quat_to_yaw
from goof_an_odd_husky_common.types import Pose2D, GpsCoord
from goof_an_odd_husky.sensor_cache import SensorCache
from goof_an_odd_husky.publisher_communicator import PublisherCommunicator


class ControllerNode(Node):
    """
    Middle-layer Node: 
    OSM Global Planning -> Corridor Generation -> Nav2 Local Execution.
    """

    def __init__(self, use_gps: bool, debug: bool = False) -> None:
        super().__init__(
            "controller_node",
            allow_undeclared_parameters=True,
            automatically_declare_parameters_from_overrides=True
        )
        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.use_gps = use_gps
        self.sensor_cache = SensorCache()
        self.publisher = PublisherCommunicator(self)
        
        self.path_manager = GlobalPathManager(
            use_gps=self.use_gps,
            logger=self.get_logger(),
            osm_relation_id=OSM_RELATION_ID,
        )
        self.goal_selector = LocalGoalSelector(
            max_trajectory_distance=MAX_TRAJECTORY_DISTANCE,
            logger=self.get_logger(),
            safety_radius=SAFETY_RADIUS,
        )

        self.nav2_client = ActionClient(self, FollowPath, 'follow_path', callback_group=MutuallyExclusiveCallbackGroup())
        self.nav2_goal_handle = None

        self.virtual_wall_pub = self.create_publisher(PointCloud2, '/virtual_corridor', 10)

        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.gps_cb_group = MutuallyExclusiveCallbackGroup()
        self.nav_cmd_cb_group = MutuallyExclusiveCallbackGroup()

        self.odom_subscription = self.create_subscription(
            Odometry, TOPICS["odom"], self.odom_callback, 10, callback_group=self.odom_cb_group
        )
        if self.use_gps:
            self.gps_subscription = self.create_subscription(
                NavSatFix, TOPICS["gps"], self.gps_callback, 10, callback_group=self.gps_cb_group
            )
            
        self.goal_subscription = self.create_subscription(
            PoseStamped, "/nav/goal", self.goal_callback, LATCHED_QOS, callback_group=self.nav_cmd_cb_group
        )
        self.cancel_subscription = self.create_subscription(
            Header, "/nav/cancel", self.cancel_callback, LATCHED_QOS, callback_group=self.nav_cmd_cb_group
        )
        self.heartbeat_subscription = self.create_subscription(
            Empty, "/viz/heartbeat", self.heartbeat_callback, 10, callback_group=self.nav_cmd_cb_group
        )
        self.nav2_vel_sub = self.create_subscription(
            Twist, 
            '/husky/nav2_cmd_vel', 
            self.nav2_vel_callback, 
            10
        )

        self.control_timer = self.create_timer(
            0.1, self.visualizer_update_loop, callback_group=self.control_callback_group
        )

        self.publisher.publish_status("idle")
        self.get_logger().info("OSM-to-Nav2 Controller Node Initialized.")

    def goal_callback(self, msg: PoseStamped) -> None:
        """Translates goal to OSM Path, and hands it to Nav2."""
        self.last_goal_stamp = msg.header.stamp
        
        odom, _, _, gps = self.sensor_cache.get_core_sensors()
        if not odom or (self.use_gps and not gps):
            self.get_logger().error("Cannot plan: missing odometry or GPS anchor!")
            return
        current_frame_id = odom.header.frame_id

        o = odom.pose.pose.orientation
        robot_x = odom.pose.pose.position.x
        robot_y = odom.pose.pose.position.y

        self.get_logger().info("========== OSM DEBUG ==========")
        self.get_logger().info(f"1. Target GPS received: Lat={msg.pose.position.x:.6f}, Lon={msg.pose.position.y:.6f}")
        self.get_logger().info(f"2. Robot current GPS: Lat={gps.lat:.6f}, Lon={gps.lon:.6f}")
        self.get_logger().info(f"3. Robot current Odom: X={robot_x:.2f}, Y={robot_y:.2f}")

        self.publisher.publish_status("computing_path")

        gps_coord = GpsCoord(msg.pose.position.x, msg.pose.position.y)
        self.path_manager.set_goal(gps_coord)
        
        self.get_logger().info("4. Calling path_manager.generate_path()...")
        try:
            self.path_manager.generate_path(gps, robot_x, robot_y, MAX_PATH_EDGE)
            
            self.path_manager.update_local_path_from_gps(gps, robot_x, robot_y)
            
        except Exception as e:
            self.get_logger().error(f"EXCEPTION inside generate_path: {e}")
            self.publisher.publish_status("error")
            return

        if not self.path_manager.has_goal():
            self.get_logger().error("5. path_manager.has_goal() returned False! (A* likely failed to find a route)")
        
        path_snapshot = self.path_manager.get_local_path_snapshot()
        self.get_logger().info(f"6. Local path snapshot length: {len(path_snapshot) if path_snapshot else 'None'}")
        self.get_logger().info("===============================")

        if not path_snapshot:
            self.get_logger().error("OSM Global Path generation failed! (Snapshot is empty)")
            self.publisher.publish_status("idle")
            return

        self.publisher.publish_global_path(path_snapshot)
        self.send_path_to_nav2(path_snapshot, current_frame_id)

    def send_path_to_nav2(self, path_snapshot: list[tuple[float, float]], frame_id: str) -> None:
        if not self.nav2_client.wait_for_server(timeout_sec=5.0):
            self.get_logger().error("Nav2 FollowPath server not found!")
            self.publisher.publish_status("error")
            return

        ros_path = Path()
        ros_path.header.frame_id = frame_id
        ros_path.header.frame_id = "husky/map"
        ros_path.header.stamp = self.get_clock().now().to_msg()
        
        for i in range(len(path_snapshot)):
            x, y = path_snapshot[i]
            pose = PoseStamped()
            pose.header = ros_path.header
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)

            yaw = 0.0
            if i < len(path_snapshot) - 1:
                nx, ny = path_snapshot[i + 1]
                yaw = math.atan2(ny - y, nx - x)
            elif i > 0:
                px, py = path_snapshot[i - 1]
                yaw = math.atan2(y - py, x - px)

            pose.pose.orientation.z = math.sin(yaw / 2.0)
            pose.pose.orientation.w = math.cos(yaw / 2.0)
            
            ros_path.poses.append(pose)

        goal_msg = FollowPath.Goal()
        goal_msg.path = ros_path
        goal_msg.controller_id = 'FollowPath'

        future = self.nav2_client.send_goal_async(goal_msg)
        future.add_done_callback(self.nav2_goal_response_callback)

    def nav2_goal_response_callback(self, future):
        goal_handle = future.result()
        if not goal_handle.accepted:
            self.get_logger().error("Nav2 rejected OSM path!")
            return
        self.nav2_goal_handle = goal_handle
        self.publisher.publish_status("navigating")
        goal_handle.get_result_async().add_done_callback(self.nav2_result_callback)

    def nav2_result_callback(self, future):
        self.nav2_goal_handle = None
        self.publisher.publish_status("goal_reached")

    def nav2_vel_callback(self, msg: Twist) -> None:
        self.get_logger().info(
            f"BRIDGE CAUGHT VELOCITY -> FWD: {msg.linear.x:.2f} | TURN: {msg.angular.z:.2f}", 
            throttle_duration_sec=1.0
        )
        
        self.publisher.publish_velocity(
            msg.linear.x, 
            msg.angular.z, 
            self.get_clock().now().to_msg()
        )

    def cancel_callback(self, msg: Header) -> None:
        self.path_manager.cancel_goal()
        if self.nav2_goal_handle:
            self.nav2_goal_handle.cancel_goal_async()
            self.nav2_goal_handle = None
        self.publisher.publish_status("idle")

    def _convert_barriers_to_pc2(self, barriers: list, robot_pose: Pose2D) -> PointCloud2:
        """Converts geometric LineObstacles to a PointCloud2 message in the robot's local frame."""
        points =[]
        resolution = 0.1 

        cos_t = math.cos(-robot_pose.theta)
        sin_t = math.sin(-robot_pose.theta)

        for b in barriers:
            if hasattr(b, 'p1') and hasattr(b, 'p2'):
                x1, y1 = float(b.p1[0]), float(b.p1[1])
                x2, y2 = float(b.p2[0]), float(b.p2[1])
            elif hasattr(b, 'start') and hasattr(b, 'end'):
                x1, y1 = float(b.start[0]), float(b.start[1])
                x2, y2 = float(b.end[0]), float(b.end[1])
            elif hasattr(b, 'x1'):
                x1, y1, x2, y2 = float(b.x1), float(b.y1), float(b.x2), float(b.y2)
            else:
                continue

            dist = math.hypot(x2 - x1, y2 - y1)
            num_points = max(int(dist / resolution), 1)
            
            for i in range(num_points + 1):
                t = i / num_points
                px = x1 + t * (x2 - x1)
                py = y1 + t * (y2 - y1)
                
                dx = px - robot_pose.x
                dy = py - robot_pose.y
                local_x = dx * cos_t - dy * sin_t
                local_y = dx * sin_t + dy * cos_t
                
                points.append([local_x, local_y, 0.0]) 

        header = Header()
        header.frame_id = "husky/base_link"
        header.stamp = self.get_clock().now().to_msg()
        
        return pc2.create_cloud_xyz32(header, points)

    def visualizer_update_loop(self) -> None:
        """Main 10Hz loop for UI updates and Virtual Wall generation."""
        odom, _, _, _ = self.sensor_cache.get_core_sensors()
        now = self.get_clock().now()

        if not self.sensor_cache.is_visualizer_alive(now, HEARTBEAT_TIMEOUT_SEC):
            if self.nav2_goal_handle:
                self.get_logger().error("Visualizer Timeout! Stopping Robot.")
                self.nav2_goal_handle.cancel_goal_async()
                self.nav2_goal_handle = None
                self.publisher.publish_status("e_stop: viz_timeout")
            return

        if odom:
            current_frame_id = odom.header.frame_id
            o = odom.pose.pose.orientation
            robot_pose = Pose2D(odom.pose.pose.position.x, odom.pose.pose.position.y, quat_to_yaw(o.x, o.y, o.z, o.w))
            self.publisher.publish_robot_pose(robot_pose)

            path_snap = self.path_manager.get_local_path_snapshot()
            if self.nav2_goal_handle and path_snap:
                selection = self.goal_selector.select_local_goal(
                    path=path_snap,
                    start_index=self.path_manager.get_current_index(),
                    vehicle_x=robot_pose.x, vehicle_y=robot_pose.y,
                    yaw=robot_pose.theta, detected_obstacles=[]
                )
                if selection:
                    _, new_idx, target_idx = selection
                    self.path_manager.update_current_index(new_idx)
                    
                    corridor = self.goal_selector.generate_corridor_barriers(
                        path=path_snap, closest_idx=new_idx, target_idx=target_idx,
                        vehicle_x=robot_pose.x, vehicle_y=robot_pose.y,
                        yaw=robot_pose.theta, barrier_offset=BARRIER_OFFSET
                    )
                    
                    all_barriers = corridor[0] + corridor[1]
                    
                    pc2_msg = self._convert_barriers_to_pc2(all_barriers, robot_pose=robot_pose)
                    # self.virtual_wall_pub.publish(pc2_msg)

            self.publisher.publish_obstacles(None)
            self.publisher.publish_trajectory(None)

    def heartbeat_callback(self, _: Empty) -> None:
        self.sensor_cache.update_heartbeat(self.get_clock().now())

    def odom_callback(self, msg: Odometry) -> None:
        self.sensor_cache.update_odom(msg)

    def gps_callback(self, msg: NavSatFix) -> None:
        self.sensor_cache.update_gps(GpsCoord(msg.latitude, msg.longitude), msg.status.status >= 0)

def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode(use_gps=USE_GPS, debug=DEBUG)
    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(node)
    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()

if __name__ == "__main__":
    main()
