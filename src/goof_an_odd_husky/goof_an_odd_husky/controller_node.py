import json
import time
import math
import threading

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.transform import Rotation

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
    QoSDurabilityPolicy,
    QoSProfile,
    QoSReliabilityPolicy,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped
from std_msgs.msg import String, Empty

from goof_an_odd_husky.config import (
    TOPICS,
    OSM_RELATION_ID,
    MAX_TRAJECTORY_DISTANCE,
)
from goof_an_odd_husky_common.obstacles import (
    CircleObstacle,
    LineObstacle,
)
from goof_an_odd_husky.helpers import gps_to_vector
from goof_an_odd_husky.local_navigation.teb_planner import TEBPlanner
from goof_an_odd_husky.local_navigation.trajectory_planner import TrajectoryPlanner
from goof_an_odd_husky.local_navigation.obstacle_pipeline import ObstaclePipeline
from goof_an_odd_husky.local_navigation.kinematics import get_odom_delta
from goof_an_odd_husky.global_navigation.graph import (
    load_graph_for_relation,
    filter_walkable_paved,
)
from goof_an_odd_husky.global_navigation.routing import (
    path_between_coordinates,
    stitch_path_coords,
    slice_path,
)
from goof_an_odd_husky.action import trajectory_to_action
from goof_an_odd_husky_msgs.msg import ObstacleArray
from goof_an_odd_husky_common.obstacles import obstacles_to_msg

LATCHED_QOS = QoSProfile(
    depth=1,
    durability=QoSDurabilityPolicy.TRANSIENT_LOCAL,
    reliability=QoSReliabilityPolicy.RELIABLE,
)
HEARTBEAT_TIMEOUT_SEC = 5.0


class ControllerNode(Node):
    control_callback_group: MutuallyExclusiveCallbackGroup
    velocity_publisher: rclpy.publisher.Publisher
    lidar_subscription: rclpy.subscription.Subscription
    gps_subscription: rclpy.subscription.Subscription | None
    odom_subscription: rclpy.subscription.Subscription
    control_timer: rclpy.timer.Timer
    data_lock: threading.Lock
    latest_odom: Odometry | None
    latest_scan: LaserScan | None
    last_scan_time: rclpy.time.Time | None
    latest_gps: NavSatFix | None
    last_processed_odom: Odometry | None
    first_gps: NavSatFix | None
    last_heartbeat_time: rclpy.time.Time | None
    planner: TrajectoryPlanner
    obstacle_pipeline: ObstaclePipeline
    initial_start: list[float]
    initial_goal: list[float]
    goal_reached: bool
    goal_lat_lon: tuple[float, float] | None
    needs_initial_plan: bool
    use_gps: bool

    def __init__(self, debug: bool = False, use_gps: bool = True) -> None:
        super().__init__("controller_node")
        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.use_gps = use_gps
        self.data_lock = threading.Lock()

        self.control_callback_group = MutuallyExclusiveCallbackGroup()
        self.lidar_cb_group = MutuallyExclusiveCallbackGroup()
        self.odom_cb_group = MutuallyExclusiveCallbackGroup()
        self.gps_cb_group = MutuallyExclusiveCallbackGroup()
        self.nav_cmd_cb_group = MutuallyExclusiveCallbackGroup()

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            TOPICS["scan"],
            self.lidar_callback,
            qos_profile_sensor_data,
            callback_group=self.lidar_cb_group,
        )
        self.odom_subscription = self.create_subscription(
            Odometry,
            TOPICS["odom"],
            self.odom_callback,
            10,
            callback_group=self.odom_cb_group,
        )
        self.gps_subscription = None
        if self.use_gps:
            self.gps_subscription = self.create_subscription(
                NavSatFix,
                TOPICS["gps"],
                self.gps_callback,
                10,
                callback_group=self.gps_cb_group,
            )

        self.goal_subscription = self.create_subscription(
            PoseStamped,
            "/nav/goal",
            self.goal_callback,
            LATCHED_QOS,
            callback_group=self.nav_cmd_cb_group,
        )
        self.cancel_subscription = self.create_subscription(
            Empty,
            "/nav/cancel",
            self.cancel_callback,
            10,
            callback_group=self.nav_cmd_cb_group,
        )
        self.heartbeat_subscription = self.create_subscription(
            Empty,
            "/viz/heartbeat",
            self.heartbeat_callback,
            10,
            callback_group=self.nav_cmd_cb_group,
        )

        self.velocity_publisher = self.create_publisher(
            TwistStamped, TOPICS["cmd_vel"], 10
        )
        self.pose_publisher = self.create_publisher(PoseStamped, "/viz/robot_pose", 10)
        self.trajectory_publisher = self.create_publisher(Path, "/viz/trajectory", 10)
        self.global_path_publisher = self.create_publisher(
            Path, "/viz/global_path", LATCHED_QOS
        )
        self.obstacles_publisher = self.create_publisher(
            ObstacleArray, "/viz/obstacles", 10
        )
        self.goal_local_publisher = self.create_publisher(
            PointStamped, "/viz/goal_local", 10
        )
        self.status_publisher = self.create_publisher(
            String, "/nav/status", LATCHED_QOS
        )

        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.control_callback_group
        )

        self.latest_odom = None
        self.latest_scan = None
        self.last_scan_time = None
        self.latest_gps = None
        self.last_processed_odom = None
        self.first_gps = None
        self.last_heartbeat_time = None

        self.initial_start = [0.0, 0.0, 0.0]
        self.initial_goal = [0.0, 0.0, 0.0]
        self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)

        self.goal_reached = False
        self.goal_lat_lon = None
        self.needs_initial_plan = False

        self.needs_global_path = False
        self.global_path_local = []
        self.global_path_lat_lon = []
        self.current_path_index = 0

        if self.use_gps:
            self.graph = filter_walkable_paved(load_graph_for_relation(OSM_RELATION_ID))

        self.obstacle_pipeline = ObstaclePipeline(
            cluster_break_distance=2, geometry_split_threshold=3
        )

        self._publish_status("idle")

    def heartbeat_callback(self, _: Empty) -> None:
        with self.data_lock:
            self.last_heartbeat_time = self.get_clock().now()

    def goal_callback(self, msg: PoseStamped) -> None:
        with self.data_lock:
            self.goal_lat_lon = (msg.pose.position.x, msg.pose.position.y)
            self.goal_reached = False
            self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)
            self.needs_initial_plan = True
            self.needs_global_path = True
            self.global_path_local = []
            self.global_path_lat_lon = []
            self.current_path_index = 0
        self._publish_status("navigating")
        self.get_logger().info(
            f"New goal set: {msg.pose.position.x}, {msg.pose.position.y}"
        )

    def cancel_callback(self, _: Empty) -> None:
        with self.data_lock:
            self.goal_lat_lon = None
            self.goal_reached = False
            self.needs_initial_plan = False
            self.needs_global_path = False
            self.global_path_local = []
            self.global_path_lat_lon = []
            self.current_path_index = 0
        self._publish_velocity(0.0, 0.0)
        self._publish_status("idle")
        self.get_logger().info("Navigation cancelled")

    def lidar_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.latest_scan, self.last_scan_time = msg, self.get_clock().now()

    def gps_callback(self, msg: NavSatFix) -> None:
        with self.data_lock:
            self.latest_gps = msg
            if not self.first_gps and msg.status.status >= 0:
                self.first_gps = msg

    def odom_callback(self, msg: Odometry) -> None:
        with self.data_lock:
            self.latest_odom = msg

    def _visualizer_is_alive(self) -> bool:
        with self.data_lock:
            if self.last_heartbeat_time is None:
                return False
            elapsed = (
                self.get_clock().now() - self.last_heartbeat_time
            ).nanoseconds / 1e9
            return elapsed < HEARTBEAT_TIMEOUT_SEC

    def control_loop(self) -> None:
        start_time = time.perf_counter()
        performance = {}

        with self.data_lock:
            scan, scan_time = self.latest_scan, self.last_scan_time
            odom_g = self.latest_odom
            gps_data = self.latest_gps
            goal_lat_lon = self.goal_lat_lon

        t1 = time.perf_counter()
        performance["Data acquisition"] = round((t1 - start_time) * 1000, 2)

        if scan is None or odom_g is None or (self.use_gps and gps_data is None):
            missing = []
            if scan is None:
                missing.append("Lidar")
            if odom_g is None:
                missing.append("Odom")
            if self.use_gps and gps_data is None:
                missing.append("GPS")
            self.get_logger().warn(
                f"Waiting for sensors ({', '.join(missing)})...",
                throttle_duration_sec=2.0,
            )
            return

        self._update_odometry(odom_g)

        orientation = odom_g.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("zyx")[0]
        robot_pose = [odom_g.pose.pose.position.x, odom_g.pose.pose.position.y, yaw]

        self._log_robot_location(odom_g, gps_data, yaw)

        if not self._visualizer_is_alive():
            self.get_logger().warn(
                "Waiting for visualizer...", throttle_duration_sec=2.0
            )
            self._publish_velocity(0.0, 0.0)
            return

        if self.use_gps and getattr(self, "global_path_lat_lon", None):
            updated_path = []
            for lat, lon in self.global_path_lat_lon:
                dx, dy = gps_to_vector(gps_data.latitude, gps_data.longitude, lat, lon)
                updated_path.append(
                    (odom_g.pose.pose.position.x + dx, odom_g.pose.pose.position.y + dy)
                )
            with self.data_lock:
                self.global_path_local = updated_path
            self._publish_global_path()

        with self.data_lock:
            global_path_snapshot = list(self.global_path_local)

        detected_obstacles = self._process_obstacles(scan, scan_time)
        t2 = time.perf_counter()
        performance["Obstacle processing"] = round((t2 - t1) * 1000, 2)

        if detected_obstacles is None:
            self._publish_velocity(0.0, 0.0)
            return

        self._publish_obstacles(detected_obstacles)

        if goal_lat_lon is None:
            self._publish_robot_pose(robot_pose)
            self._publish_velocity(0.0, 0.0)
            performance["Total"] = round((time.perf_counter() - start_time) * 1000, 2)
            self.get_logger().debug(
                f"Idle - Performance:\n{json.dumps(performance, indent=4)}",
                throttle_duration_sec=5.0,
            )
            return

        if global_path_snapshot:
            final_x, final_y = global_path_snapshot[-1]
            dist_to_final = math.hypot(final_x - robot_pose[0], final_y - robot_pose[1])
            if dist_to_final < 1.0:
                if not self.goal_reached:
                    self.goal_reached = True
                    self.get_logger().info("Global Goal Reached!")
                    self._publish_velocity(0.0, 0.0)
                    self._publish_status("goal_reached")
                    with self.data_lock:
                        self.goal_lat_lon = None
                        self.global_path_local = []
                        self.current_path_index = 0
                return

        self.goal_reached = False

        local_goal = self._update_local_goal(
            goal_lat_lon, odom_g, gps_data, detected_obstacles
        )
        t3 = time.perf_counter()
        performance["Goal update"] = round((t3 - t2) * 1000, 2)

        if local_goal is None:
            self._publish_velocity(0.0, 0.0)
            return

        current_velocity = odom_g.twist.twist.linear.x
        current_omega = odom_g.twist.twist.angular.z
        self.planner.refine(
            current_velocity=current_velocity, current_omega=current_omega
        )

        t4 = time.perf_counter()
        performance["Planner refinement"] = round((t4 - t3) * 1000, 2)

        trajectory = self.planner.get_trajectory()

        with self.data_lock:
            latest_odom_after = self.latest_odom
        if latest_odom_after and trajectory is not None and len(trajectory) > 0:
            x1, y1, yaw1 = robot_pose
            x2, y2 = (
                latest_odom_after.pose.pose.position.x,
                latest_odom_after.pose.pose.position.y,
            )
            o2 = latest_odom_after.pose.pose.orientation
            yaw2 = Rotation.from_quat([o2.x, o2.y, o2.z, o2.w]).as_euler("zyx")[0]
            if x1 != x2 or y1 != y2 or yaw1 != yaw2:
                dx2, dy2, dtheta2 = get_odom_delta(latest_odom_after, odom_g)
                self.planner.transform_trajectory(dx2, dy2, dtheta2)
                trajectory = self.planner.get_trajectory()
                self.last_processed_odom = latest_odom_after
                robot_pose = [x2, y2, yaw2]

        v, omega = trajectory_to_action(trajectory)
        self._publish_velocity(v, omega)

        t5 = time.perf_counter()
        performance["Publish/Finish"] = round((t5 - t4) * 1000, 2)
        performance["Total"] = round((t5 - start_time) * 1000, 2)
        self.get_logger().debug(
            f"Performance:\n{json.dumps(performance, indent=4)}",
            throttle_duration_sec=2.0,
        )

        self._publish_robot_pose(robot_pose)
        self._publish_trajectory(trajectory)
        self._publish_goal_local(local_goal)

    def _log_robot_location(
        self, odom_g: Odometry, gps_data: NavSatFix | None, yaw: float
    ) -> None:
        """Logs location."""
        if self.use_gps and gps_data:
            log_msg = f"GPS: Lat={gps_data.latitude:.6f}, Lon={gps_data.longitude:.6f}, Yaw={yaw:.2f}rad"
        else:
            log_msg = f"Pose: X={odom_g.pose.pose.position.x:.2f}m, Y={odom_g.pose.pose.position.y:.2f}m, Yaw={yaw:.2f}rad"

        self.get_logger().info(log_msg, throttle_duration_sec=2.0)

    def _generate_global_path(
        self, goal_lat_lon: tuple, gps_data: NavSatFix, odom_g: Odometry
    ) -> None:
        try:
            path_coords_local = []
            if self.use_gps:
                self.get_logger().info("Generating A* global path from OSM graph...")
                path_nodes, G_ext = path_between_coordinates(
                    self.graph,
                    gps_data.latitude,
                    gps_data.longitude,
                    goal_lat_lon[0],
                    goal_lat_lon[1],
                )
                path_coords = stitch_path_coords(G_ext, path_nodes)
                sliced_path = slice_path(path_coords)

                with self.data_lock:
                    self.global_path_lat_lon = sliced_path
            else:
                self.get_logger().info(
                    "Generating straight line global path (No GPS)..."
                )
                start_x = odom_g.pose.pose.position.x
                start_y = odom_g.pose.pose.position.y
                dx = goal_lat_lon[0] - start_x
                dy = goal_lat_lon[1] - start_y

                dist = math.hypot(dx, dy)
                slices = max(1, int(dist / 2.0))
                for i in range(slices + 1):
                    t = i / slices
                    path_coords_local.append((start_x + t * dx, start_y + t * dy))

            with self.data_lock:
                self.global_path_local = path_coords_local
                self.current_path_index = 0
                self.needs_global_path = False

            self._publish_global_path()

        except Exception as e:
            self.get_logger().error(f"Failed to generate global path: {e}")
            with self.data_lock:
                self.needs_global_path = True

    def _is_point_safe(
        self, local_x: float, local_y: float, obstacles: list, margin: float = 0.4
    ) -> bool:
        """Checks if a point in the base_link frame is too close to any obstacle."""
        if not obstacles:
            return True

        for obs in obstacles:
            if isinstance(obs, CircleObstacle):
                if math.hypot(local_x - obs.x, local_y - obs.y) < (obs.radius + margin):
                    return False
            elif isinstance(obs, LineObstacle):
                l2 = (obs.x2 - obs.x1) ** 2 + (obs.y2 - obs.y1) ** 2
                if l2 == 0:
                    dist = math.hypot(local_x - obs.x1, local_y - obs.y1)
                else:
                    t = max(
                        0,
                        min(
                            1,
                            (
                                (local_x - obs.x1) * (obs.x2 - obs.x1)
                                + (local_y - obs.y1) * (obs.y2 - obs.y1)
                            )
                            / l2,
                        ),
                    )
                    proj_x = obs.x1 + t * (obs.x2 - obs.x1)
                    proj_y = obs.y1 + t * (obs.y2 - obs.y1)
                    dist = math.hypot(local_x - proj_x, local_y - proj_y)

                if dist < margin:
                    return False
        return True

    def _update_local_goal(
        self,
        goal_lat_lon: tuple,
        odom_g: Odometry,
        gps_data: NavSatFix,
        detected_obstacles: list,
    ) -> list[float] | None:
        if self.use_gps and not self.first_gps:
            self.get_logger().warn(
                "Waiting for GPS anchor...", throttle_duration_sec=2.0
            )
            return None

        if self.needs_global_path:
            self._generate_global_path(goal_lat_lon, gps_data, odom_g)

        with self.data_lock:
            path_snapshot = list(self.global_path_local)
            start_index = self.current_path_index

        if not path_snapshot:
            return None

        vehicle_x = odom_g.pose.pose.position.x
        vehicle_y = odom_g.pose.pose.position.y
        orientation = odom_g.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("zyx")[0]

        min_dist = float("inf")
        closest_idx = max(start_index - 3, 0)

        search_window_end = min(len(path_snapshot), start_index + 8)
        for i in range(start_index, search_window_end):
            px, py = path_snapshot[i]
            dist = math.hypot(px - vehicle_x, py - vehicle_y)
            if dist < min_dist:
                min_dist = dist
                closest_idx = i

        with self.data_lock:
            self.current_path_index = closest_idx

        target_idx = closest_idx
        for i in range(closest_idx, len(path_snapshot)):
            px, py = path_snapshot[i]
            dist_from_robot = math.hypot(px - vehicle_x, py - vehicle_y)
            if dist_from_robot >= MAX_TRAJECTORY_DISTANCE:
                target_idx = i
                break
        else:
            target_idx = len(path_snapshot) - 1

        if target_idx < 0 or target_idx >= len(path_snapshot):
            return None

        c, s = np.cos(-yaw), np.sin(-yaw)

        def get_local_coords(idx):
            gx, gy = path_snapshot[idx]
            dx, dy = gx - vehicle_x, gy - vehicle_y
            return dx * c - dy * s, dx * s + dy * c

        local_x, local_y = get_local_coords(target_idx)

        if not self._is_point_safe(local_x, local_y, detected_obstacles, margin=2.0):
            max_search_offset = 8
            found_safe = False

            for offset in range(1, max_search_offset + 1):
                idx_ahead = target_idx + offset
                if idx_ahead < len(path_snapshot):
                    lx, ly = get_local_coords(idx_ahead)
                    if self._is_point_safe(lx, ly, detected_obstacles, margin=0.4):
                        local_x, local_y = lx, ly
                        found_safe = True
                        break

                idx_behind = target_idx - offset
                if idx_behind >= closest_idx:
                    lx, ly = get_local_coords(idx_behind)
                    if self._is_point_safe(lx, ly, detected_obstacles, margin=0.4):
                        local_x, local_y = lx, ly
                        found_safe = True
                        break

            if not found_safe:
                self.get_logger().debug(
                    "Could not slide target point out of obstacle. Reverting to closest node.",
                    throttle_duration_sec=2.0,
                )
                local_x, local_y = get_local_coords(closest_idx)

        self.planner.move_goal((local_x, local_y), (0.0,))

        if self.needs_initial_plan:
            self.planner.plan()
            self.needs_initial_plan = False

        return [local_x, local_y, 0.0]

    def _update_odometry(self, odom_g: Odometry) -> None:
        dx, dy, dt = get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

    def _process_obstacles(self, scan: LaserScan, scan_time) -> list | None:
        if (self.get_clock().now() - scan_time).nanoseconds / 1e9 > 1.0:
            self.get_logger().error("Lidar data stale", throttle_duration_sec=1.0)
            return None
        obstacles = self.obstacle_pipeline.process(scan)
        self.planner.update_obstacles(obstacles)
        return obstacles

    def _publish_velocity(self, v: float, omega: float) -> None:
        msg = TwistStamped()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(omega)
        self.velocity_publisher.publish(msg)

    def _publish_status(self, status: str) -> None:
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

    def _publish_robot_pose(self, robot_pose: list[float]) -> None:
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"
        msg.pose.position.x, msg.pose.position.y = robot_pose[0], robot_pose[1]
        q = Rotation.from_euler("z", robot_pose[2]).as_quat()
        msg.pose.orientation.x, msg.pose.orientation.y = q[0], q[1]
        msg.pose.orientation.z, msg.pose.orientation.w = q[2], q[3]
        self.pose_publisher.publish(msg)

    def _publish_trajectory(self, trajectory: NDArray[np.float64] | None) -> None:
        if trajectory is None or len(trajectory) == 0:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        for point in trajectory:
            pose = PoseStamped()
            pose.pose.position.x, pose.pose.position.y = (
                float(point[0]),
                float(point[1]),
            )
            q = Rotation.from_euler("z", float(point[2])).as_quat()
            pose.pose.orientation.x, pose.pose.orientation.y = q[0], q[1]
            pose.pose.orientation.z, pose.pose.orientation.w = q[2], q[3]
            msg.poses.append(pose)
        self.trajectory_publisher.publish(msg)

    def _publish_global_path(self) -> None:
        with self.data_lock:
            path_snapshot = list(self.global_path_local)

        if not path_snapshot:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        for x, y in path_snapshot:
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.global_path_publisher.publish(msg)

    def _publish_obstacles(self, obstacles: list) -> None:
        self.obstacles_publisher.publish(obstacles_to_msg(obstacles))

    def _publish_goal_local(self, goal_local: list[float]) -> None:
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y = float(goal_local[0]), float(goal_local[1])
        self.goal_local_publisher.publish(msg)


def main(args=None) -> None:
    rclpy.init(args=args)
    node = ControllerNode(debug=True, use_gps=True)

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
