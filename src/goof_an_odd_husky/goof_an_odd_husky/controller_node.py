import json
import time
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

from goof_an_odd_husky.config import TOPICS
from goof_an_odd_husky.helpers import gps_to_vector
from goof_an_odd_husky.local_navigation.teb_planner import TEBPlanner
from goof_an_odd_husky.local_navigation.trajectory_planner import TrajectoryPlanner
from goof_an_odd_husky.local_navigation.obstacle_pipeline import ObstaclePipeline
from goof_an_odd_husky.local_navigation.kinematics import get_odom_delta
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
        self._publish_status("navigating")
        self.get_logger().info(
            f"New goal set: {'lat=' if self.use_gps else 'x='}{msg.pose.position.x}, "
            f"{'lon=' if self.use_gps else 'y='}{msg.pose.position.y}"
        )

    def cancel_callback(self, _: Empty) -> None:
        with self.data_lock:
            self.goal_lat_lon = None
            self.goal_reached = False
            self.needs_initial_plan = False
        self._publish_velocity(0.0, 0.0)
        self._publish_status("idle")
        self.get_logger().info("Navigation cancelled")

    def lidar_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.latest_scan = msg
            self.last_scan_time = self.get_clock().now()
        self.get_logger().debug("Lidar data received", throttle_duration_sec=5.0)

    def gps_callback(self, msg: NavSatFix) -> None:
        with self.data_lock:
            self.latest_gps = msg
            if not self.first_gps and msg.status.status >= 0:
                self.first_gps = msg
        self.get_logger().debug(
            f"GPS data received: {msg.latitude, msg.longitude, msg.altitude}",
            throttle_duration_sec=1.0,
        )

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

        if not self._visualizer_is_alive():
            self.get_logger().warn(
                "Waiting for visualizer...", throttle_duration_sec=2.0
            )
            self._publish_velocity(0.0, 0.0)
            return

        with self.data_lock:
            scan, scan_time = self.latest_scan, self.last_scan_time
            odom_g = self.latest_odom
            gps_data = self.latest_gps
            goal_lat_lon = self.goal_lat_lon

        t1 = time.perf_counter()
        performance["Data acquisition"] = round((t1 - start_time) * 1000, 2)

        if scan is None:
            self.get_logger().warn("Waiting for LIDAR...", throttle_duration_sec=2.0)
            return
        if odom_g is None:
            self.get_logger().warn(
                "Waiting for Fused Odometry...", throttle_duration_sec=2.0
            )
            return
        if self.use_gps and gps_data is None:
            self.get_logger().warn("Waiting for GPS...", throttle_duration_sec=2.0)
            return

        self._update_odometry(odom_g)
        current_velocity = odom_g.twist.twist.linear.x
        current_omega = odom_g.twist.twist.angular.z

        orientation = odom_g.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("zyx")[0]

        if not self.use_gps:
            self.get_logger().info(
                f"Robot Location: X={odom_g.pose.pose.position.x:.2f}m, "
                f"Y={odom_g.pose.pose.position.y:.2f}m, Yaw={yaw:.2f}rad",
                throttle_duration_sec=2.0,
            )

        robot_pose = [odom_g.pose.pose.position.x, odom_g.pose.pose.position.y, yaw]

        detected_obstacles = self._process_obstacles(scan, scan_time)
        t2 = time.perf_counter()
        performance["Obstacle processing"] = round((t2 - t1) * 1000, 2)

        if detected_obstacles is None:
            self._publish_velocity(0.0, 0.0)
            return

        self._publish_obstacles(detected_obstacles)

        if goal_lat_lon is None:
            self.get_logger().debug("Waiting for goal...", throttle_duration_sec=2.0)
            self._publish_robot_pose(robot_pose)
            performance["Total"] = round((time.perf_counter() - start_time) * 1000, 2)
            self.get_logger().debug(
                "Performance:\n" + json.dumps(performance, indent=4),
                throttle_duration_sec=2.0,
            )
            self._publish_velocity(0.0, 0.0)
            return

        local_goal = self._update_local_goal(goal_lat_lon, odom_g)
        t3 = time.perf_counter()
        performance["Goal update"] = round((t3 - t2) * 1000, 2)

        if local_goal is None:
            self._publish_velocity(0.0, 0.0)
            return

        if self.planner.get_distance_goal() < 1.0:
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info("Goal Reached")
                self._publish_velocity(0.0, 0.0)
                self._publish_status("goal_reached")
                with self.data_lock:
                    self.goal_lat_lon = None
            return

        self.goal_reached = False

        self.planner.refine(
            current_velocity=current_velocity, current_omega=current_omega
        )
        t4 = time.perf_counter()
        performance["Planner refinement"] = round((t4 - t3) * 1000, 2)

        trajectory = self.planner.get_trajectory()

        with self.data_lock:
            latest_odom_after = self.latest_odom

        if (
            latest_odom_after is not None
            and trajectory is not None
            and len(trajectory) > 0
        ):
            x1, y1, yaw1 = robot_pose

            x2 = latest_odom_after.pose.pose.position.x
            y2 = latest_odom_after.pose.pose.position.y
            orient2 = latest_odom_after.pose.pose.orientation
            yaw2 = Rotation.from_quat(
                [orient2.x, orient2.y, orient2.z, orient2.w]
            ).as_euler("zyx")[0]

            if x1 != x2 or y1 != y2 or yaw1 != yaw2:
                c1, s1 = np.cos(yaw1), np.sin(yaw1)
                traj_xg = x1 + trajectory[:, 0] * c1 - trajectory[:, 1] * s1
                traj_yg = y1 + trajectory[:, 0] * s1 + trajectory[:, 1] * c1
                traj_thetag = trajectory[:, 2] + yaw1

                c2, s2 = np.cos(yaw2), np.sin(yaw2)
                dx, dy = traj_xg - x2, traj_yg - y2
                trajectory[:, 0] = dx * c2 + dy * s2
                trajectory[:, 1] = -dx * s2 + dy * c2
                trajectory[:, 2] = traj_thetag - yaw2

                robot_pose = [x2, y2, yaw2]

        v, omega = trajectory_to_action(trajectory)
        self._publish_velocity(v, omega)

        t5 = time.perf_counter()
        performance["Publish/Finish"] = round((t5 - t4) * 1000, 2)
        performance["Total"] = round((t5 - start_time) * 1000, 2)
        self.get_logger().debug(
            "Performance:\n" + json.dumps(performance, indent=4),
            throttle_duration_sec=2.0,
        )

        self._publish_robot_pose(robot_pose)
        self._publish_trajectory(trajectory)
        self._publish_goal_local(local_goal)

    def _update_odometry(self, odom_g: Odometry) -> None:
        dx, dy, dt = get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

    def _update_local_goal(
        self, goal_lat_lon: tuple, odom_g: Odometry
    ) -> list[float] | None:
        if self.use_gps:
            if not self.first_gps:
                self.get_logger().warn(
                    "Waiting for GPS anchor...", throttle_duration_sec=2.0
                )
                return None
            global_x, global_y = gps_to_vector(
                self.first_gps.latitude,
                self.first_gps.longitude,
                goal_lat_lon[0],
                goal_lat_lon[1],
            )
        else:
            global_x, global_y = goal_lat_lon[0], goal_lat_lon[1]

        vehicle_x = odom_g.pose.pose.position.x
        vehicle_y = odom_g.pose.pose.position.y
        orientation = odom_g.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("zyx")[0]

        dx, dy = global_x - vehicle_x, global_y - vehicle_y
        c, s = np.cos(-yaw), np.sin(-yaw)
        local_x, local_y = dx * c - dy * s, dx * s + dy * c

        self.planner.move_goal((local_x, local_y), (0.0,))

        if self.needs_initial_plan:
            self.planner.plan()
            self.needs_initial_plan = False

        return [local_x, local_y, 0.0]

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
    node = ControllerNode(debug=True, use_gps=False)

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
