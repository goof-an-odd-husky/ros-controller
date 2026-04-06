import threading

import rclpy
from rclpy.node import Node
from rclpy.qos import (
    qos_profile_sensor_data,
)
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import Odometry, Path
from geometry_msgs.msg import TwistStamped, PoseStamped, PointStamped
from std_msgs.msg import String, Empty

from goof_an_odd_husky_common.config import (
    TOPICS,
    MAX_TRAJECTORY_DISTANCE,
    BARRIER_OFFSET,
    HEARTBEAT_TIMEOUT_SEC,
    USE_GPS,
    DEBUG,
)
from goof_an_odd_husky_common.helpers import (
    quat_to_yaw,
    yaw_to_quat,
)
from goof_an_odd_husky.performance_tracker import PerformanceTracker
from goof_an_odd_husky_common.types import (
    Pose2D,
    Trajectory,
    GpsCoord,
)
from goof_an_odd_husky_common.obstacles import Obstacle
from goof_an_odd_husky_common.qos import LATCHED_QOS
from goof_an_odd_husky.local_navigation.teb_planner import TEBPlanner
from goof_an_odd_husky.local_navigation.obstacle_pipeline import ObstaclePipeline
from goof_an_odd_husky.local_navigation.kinematics import get_odom_delta
from goof_an_odd_husky.local_navigation.local_goal_selector import LocalGoalSelector
from goof_an_odd_husky.global_navigation.path_manager import GlobalPathManager
from goof_an_odd_husky.action import trajectory_to_action

from goof_an_odd_husky_msgs.msg import ObstacleArray
from goof_an_odd_husky_common.obstacles import obstacles_to_msg


def make_pose_stamped(
    x: float,
    y: float,
    yaw: float,
    stamp=None,
    frame_id: str = "",
) -> PoseStamped:
    """Construct a PoseStamped message from position and heading.

    Args:
        x: X position in meters.
        y: Y position in meters.
        yaw: Heading in radians.
        stamp: Optional ROS timestamp.
        frame_id: Optional coordinate frame identifier.

    Returns:
        PoseStamped: The populated message.
    """
    msg = PoseStamped()
    if stamp is not None:
        msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.pose.position.x = x
    msg.pose.position.y = y
    q = yaw_to_quat(yaw)
    msg.pose.orientation.x = float(q[0])
    msg.pose.orientation.y = float(q[1])
    msg.pose.orientation.z = float(q[2])
    msg.pose.orientation.w = float(q[3])
    return msg


class ControllerNode(Node):
    """Main ROS 2 Node orchestrating global and local navigation.

    Attributes:
        use_gps: Boolean indicating whether to use GPS for global localization.
        data_lock: Threading lock for synchronizing sensor callbacks.
        latest_odom: The most recently received odometry message.
        latest_scan: The most recently received laser scan message.
        last_scan_time: Timestamp of the last received laser scan.
        latest_gps: The most recently received GPS coordinate (as GpsCoord).
        last_processed_odom: The odometry message used in the last planner loop.
        first_gps: The first valid GPS anchor coordinate (as GpsCoord).
        last_heartbeat_time: Timestamp of the last visualizer heartbeat.
        initial_start: Starting pose for the TEB planner.
        initial_goal: Initial goal pose for the TEB planner.
        planner: The TEB local planner instance.
        needs_initial_plan: Boolean flag to trigger an initial TEB plan.
        path_manager: Manager for the global A* path logic.
        goal_selector: Selector for the immediate local goal.
        obstacle_pipeline: Pipeline for processing LiDAR scans into geometric obstacles.
    """

    use_gps: bool
    data_lock: threading.Lock

    control_callback_group: MutuallyExclusiveCallbackGroup
    lidar_cb_group: MutuallyExclusiveCallbackGroup
    odom_cb_group: MutuallyExclusiveCallbackGroup
    gps_cb_group: MutuallyExclusiveCallbackGroup
    nav_cmd_cb_group: MutuallyExclusiveCallbackGroup

    lidar_subscription: rclpy.subscription.Subscription
    odom_subscription: rclpy.subscription.Subscription
    gps_subscription: rclpy.subscription.Subscription | None
    goal_subscription: rclpy.subscription.Subscription
    cancel_subscription: rclpy.subscription.Subscription
    heartbeat_subscription: rclpy.subscription.Subscription

    velocity_publisher: rclpy.publisher.Publisher
    pose_publisher: rclpy.publisher.Publisher
    trajectory_publisher: rclpy.publisher.Publisher
    global_path_publisher: rclpy.publisher.Publisher
    obstacles_publisher: rclpy.publisher.Publisher
    goal_local_publisher: rclpy.publisher.Publisher
    status_publisher: rclpy.publisher.Publisher

    control_timer: rclpy.timer.Timer

    latest_odom: Odometry | None
    latest_scan: LaserScan | None
    last_scan_time: rclpy.time.Time | None
    latest_gps: GpsCoord | None
    last_processed_odom: Odometry | None
    first_gps: GpsCoord | None
    last_heartbeat_time: rclpy.time.Time | None

    initial_start: list[float]
    initial_goal: list[float]
    planner: TEBPlanner
    needs_initial_plan: bool

    path_manager: GlobalPathManager
    goal_selector: LocalGoalSelector
    obstacle_pipeline: ObstaclePipeline

    def __init__(self, debug: bool = False, use_gps: bool = True) -> None:
        """Initialize the ControllerNode.

        Args:
            debug: Whether to set the logger level to DEBUG.
            use_gps: Whether to use GPS for navigation.
        """
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
        self.needs_initial_plan = False

        self.path_manager = GlobalPathManager(
            use_gps=self.use_gps, logger=self.get_logger()
        )
        self.goal_selector = LocalGoalSelector(
            max_trajectory_distance=MAX_TRAJECTORY_DISTANCE, logger=self.get_logger()
        )
        self.obstacle_pipeline = ObstaclePipeline(
            cluster_break_distance=2, geometry_split_threshold=3
        )

        self._publish_status("idle")

    def heartbeat_callback(self, _: Empty) -> None:
        """Callback for the visualizer heartbeat to track UI connection status.

        Args:
            _: Empty message payload.
        """
        with self.data_lock:
            self.last_heartbeat_time = self.get_clock().now()

    def goal_callback(self, msg: PoseStamped) -> None:
        """Callback for receiving a new global goal.

        Args:
            msg: The PoseStamped message containing the new goal.
        """
        self.path_manager.set_goal(GpsCoord(msg.pose.position.x, msg.pose.position.y))
        with self.data_lock:
            self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)
            self.needs_initial_plan = True

        self._publish_status("navigating")
        self.get_logger().info(
            f"New goal set: {msg.pose.position.x}, {msg.pose.position.y}"
        )

    def cancel_callback(self, _: Empty) -> None:
        """Callback for cancelling the current navigation goal.

        Args:
            _: Empty message payload.
        """
        self.path_manager.cancel_goal()
        with self.data_lock:
            self.needs_initial_plan = False

        self._publish_velocity(0.0, 0.0)
        self._publish_status("idle")
        self.get_logger().info("Navigation cancelled")

    def lidar_callback(self, msg: LaserScan) -> None:
        """Callback for LiDAR scan updates.

        Args:
            msg: The incoming LaserScan message.
        """
        with self.data_lock:
            self.latest_scan, self.last_scan_time = msg, self.get_clock().now()

    def gps_callback(self, msg: NavSatFix) -> None:
        """Callback for GPS position updates.

        Converts the raw NavSatFix to GpsCoord immediately for internal use.

        Args:
            msg: The incoming NavSatFix message.
        """
        with self.data_lock:
            gps_coord = GpsCoord(msg.latitude, msg.longitude)
            self.latest_gps = gps_coord
            if not self.first_gps and msg.status.status >= 0:
                self.first_gps = gps_coord

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for global Odometry updates.

        Args:
            msg: The incoming Odometry message.
        """
        with self.data_lock:
            self.latest_odom = msg

    def _visualizer_is_alive(self) -> bool:
        """Check if the external visualizer is still active.

        Returns:
            bool: True if visualizer heartbeat was recently received, False otherwise.
        """
        with self.data_lock:
            if self.last_heartbeat_time is None:
                return False
            elapsed = (
                self.get_clock().now() - self.last_heartbeat_time
            ).nanoseconds / 1e9
            return elapsed < HEARTBEAT_TIMEOUT_SEC

    def control_loop(self) -> None:
        """Main control loop responsible for executing planner step and publishing commands."""

        with PerformanceTracker(self.get_logger(), log_level='debug', throttle_sec=2.0) as performance:

            with self.data_lock:
                scan, scan_time = self.latest_scan, self.last_scan_time
                odom_g = self.latest_odom
                gps_data = self.latest_gps

            performance.update("Data acquisition")

            if scan is None or odom_g is None or (self.use_gps and gps_data is None):
                self._log_missing_sensors(scan, odom_g, gps_data)
                return

            self._update_odometry(odom_g)

            o = odom_g.pose.pose.orientation
            robot_pose = Pose2D(
                odom_g.pose.pose.position.x,
                odom_g.pose.pose.position.y,
                quat_to_yaw(o.x, o.y, o.z, o.w),
            )

            self._log_robot_location(robot_pose, gps_data)

            if not self._visualizer_is_alive():
                self.get_logger().warn("Waiting for visualizer...", throttle_duration_sec=2.0)
                self._publish_velocity(0.0, 0.0)
                return

            self.path_manager.update_local_path_from_gps(gps_data, robot_pose.x, robot_pose.y)
            local_path_snapshot = self.path_manager.get_local_path_snapshot()
            if local_path_snapshot:
                self._publish_global_path(local_path_snapshot)

            detected_obstacles = self._process_obstacles(scan, scan_time)
            performance.update("Obstacle processing")

            if detected_obstacles is None:
                self._publish_velocity(0.0, 0.0)
                return

            self._publish_obstacles(detected_obstacles)

            if not self.path_manager.has_goal():
                self._publish_robot_pose(robot_pose)
                self._publish_velocity(0.0, 0.0)
                return

            if self.use_gps and not self.first_gps:
                self.get_logger().warn("Waiting for GPS anchor...", throttle_duration_sec=2.0)
                return

            self.path_manager.generate_path(gps_data, robot_pose.x, robot_pose.y)

            if self.path_manager.check_goal_reached(robot_pose.x, robot_pose.y):
                self._publish_velocity(0.0, 0.0)
                self._publish_status("goal_reached")
                return

            local_path_snapshot = self.path_manager.get_local_path_snapshot()
            selection = self.goal_selector.select_local_goal(
                path=local_path_snapshot,
                start_index=self.path_manager.get_current_index(),
                vehicle_x=robot_pose.x,
                vehicle_y=robot_pose.y,
                yaw=robot_pose.theta,
                detected_obstacles=detected_obstacles,
            )
            performance.update("Goal update")

            if selection is None:
                self._publish_velocity(0.0, 0.0)
                return

            local_goal, new_closest_idx, target_idx = selection
            if self.use_gps:
                corridor_line_obstacles = self.goal_selector.generate_corridor_barriers(
                    path=local_path_snapshot,
                    closest_idx=new_closest_idx,
                    target_idx=target_idx,
                    vehicle_x=robot_pose.x,
                    vehicle_y=robot_pose.y,
                    yaw=robot_pose.theta,
                    barrier_offset=BARRIER_OFFSET,
                )
                detected_obstacles.extend(corridor_line_obstacles[0])
                detected_obstacles.extend(corridor_line_obstacles[1])

            self._publish_obstacles(detected_obstacles)
            self.path_manager.update_current_index(new_closest_idx)

            performance.update("Corridor generation")

            self.planner.move_goal(local_goal, (0.0,))
            if self.needs_initial_plan:
                self.planner.plan()
                self.needs_initial_plan = False

            self.planner.refine(
                current_velocity=odom_g.twist.twist.linear.x,
                current_omega=odom_g.twist.twist.angular.z,
            )

            performance.update("Planner refinement")

            trajectory = self.planner.get_trajectory()

            with self.data_lock:
                latest_odom_after = self.latest_odom

            if latest_odom_after and trajectory is not None and len(trajectory) > 0:
                o2 = latest_odom_after.pose.pose.orientation
                x2 = latest_odom_after.pose.pose.position.x
                y2 = latest_odom_after.pose.pose.position.y
                yaw2 = quat_to_yaw(o2.x, o2.y, o2.z, o2.w)

                if robot_pose.x != x2 or robot_pose.y != y2 or robot_pose.theta != yaw2:
                    delta = get_odom_delta(latest_odom_after, odom_g)
                    self.planner.transform_trajectory(delta.dx, delta.dy, delta.dtheta)
                    trajectory = self.planner.get_trajectory()
                    self.last_processed_odom = latest_odom_after
                    robot_pose = Pose2D(x2, y2, yaw2)

            v, omega = trajectory_to_action(trajectory)
            self._publish_velocity(v, omega)

            performance.update("Publish/Finish")

            self._publish_robot_pose(robot_pose)
            self._publish_trajectory(trajectory)
            self._publish_goal_local([local_goal[0], local_goal[1], 0.0])

    def _log_missing_sensors(
        self,
        scan: LaserScan | None,
        odom: Odometry | None,
        gps: GpsCoord | None,
    ) -> None:
        """Helper to log warnings when required sensors are absent.

        Args:
            scan: Optional LaserScan message.
            odom: Optional Odometry message.
            gps: Optional GpsCoord coordinate.
        """
        missing = []
        if scan is None:
            missing.append("Lidar")
        if odom is None:
            missing.append("Odom")
        if self.use_gps and gps is None:
            missing.append("GPS")
        self.get_logger().warn(
            f"Waiting for sensors ({', '.join(missing)})...", throttle_duration_sec=2.0
        )

    def _log_robot_location(
        self, robot_pose: Pose2D, gps_data: GpsCoord | None
    ) -> None:
        """Helper to log the current robot location.

        Args:
            robot_pose: The current x,y,theta position.
            gps_data: The GpsCoord coordinate, or None.
        """
        if self.use_gps and gps_data:
            msg = (
                f"GPS: Lat={gps_data.lat:.6f}, "
                f"Lon={gps_data.lon:.6f}, "
                f"Yaw={robot_pose.theta:.2f}rad"
            )
        else:
            msg = f"Pose: X={robot_pose.x:.2f}m, Y={robot_pose.y:.2f}m, Yaw={robot_pose.theta:.2f}rad"
        self.get_logger().info(msg, throttle_duration_sec=2.0)

    def _update_odometry(self, odom_g: Odometry) -> None:
        """Updates the planner's internal trajectory state based on new odometry.

        Args:
            odom_g: The Odometry message.
        """
        dx, dy, dt = get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

    def _process_obstacles(
        self, scan: LaserScan, scan_time: rclpy.time.Time
    ) -> list[Obstacle] | None:
        """Extract geometric obstacles from the LiDAR scan.

        Args:
            scan: The LaserScan message.
            scan_time: The timestamp the scan was received.

        Returns:
            list[Obstacle] | None: A list of extracted obstacles, or None if the scan is stale.
        """
        if (self.get_clock().now() - scan_time).nanoseconds / 1e9 > 1.0:
            self.get_logger().error("Lidar data stale", throttle_duration_sec=1.0)
            return None
        obstacles = self.obstacle_pipeline.process(scan)
        self.planner.update_obstacles(obstacles)
        return obstacles

    def _publish_velocity(self, v: float, omega: float) -> None:
        """Publish the velocity command to the robot base.

        Args:
            v: Linear velocity in m/s.
            omega: Angular velocity in rad/s.
        """
        msg = TwistStamped()
        msg.twist.linear.x = float(v)
        msg.twist.angular.z = float(omega)
        self.velocity_publisher.publish(msg)

    def _publish_status(self, status: str) -> None:
        """Publish a text status indicating the robot state.

        Args:
            status: The status string.
        """
        msg = String()
        msg.data = status
        self.status_publisher.publish(msg)

    def _publish_robot_pose(self, pose: Pose2D) -> None:
        """Publish the robot's pose for visualization.

        Args:
            pose: A Pose2D containing x, y, theta.
        """
        self.pose_publisher.publish(
            make_pose_stamped(
                pose.x, pose.y, pose.theta, self.get_clock().now().to_msg(), "odom"
            )
        )

    def _publish_trajectory(self, trajectory: Trajectory | None) -> None:
        """Publish the current locally planned trajectory path.

        Args:
            trajectory: An Nx4 numpy array representing the trajectory.
        """
        if trajectory is None or len(trajectory) == 0:
            return
        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        for point in trajectory:
            msg.poses.append(
                make_pose_stamped(float(point[0]), float(point[1]), float(point[2]))
            )
        self.trajectory_publisher.publish(msg)

    def _publish_global_path(self, path_local: list[tuple[float, float]]) -> None:
        """Publish the global path shifted into the local odometry frame.

        Args:
            path_local: A list of (x, y) coordinates representing the path.
        """
        if not path_local:
            return

        msg = Path()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "odom"

        for x, y in path_local:
            pose = PoseStamped()
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.orientation.w = 1.0
            msg.poses.append(pose)

        self.global_path_publisher.publish(msg)

    def _publish_obstacles(self, obstacles: list[Obstacle]) -> None:
        """Publish detected obstacles for visualization.

        Args:
            obstacles: A list of geometric obstacles.
        """
        self.obstacles_publisher.publish(obstacles_to_msg(obstacles))

    def _publish_goal_local(self, goal_local: list[float]) -> None:
        """Publish the specific local goal currently targeted by the TEB planner.

        Args:
            goal_local: A list containing [x, y, theta].
        """
        msg = PointStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = "base_link"
        msg.point.x, msg.point.y = float(goal_local[0]), float(goal_local[1])
        self.goal_local_publisher.publish(msg)


def main(args=None) -> None:
    """Entry point for running the controller node.

    Args:
        args: Command line arguments passed to ROS 2 init.
    """
    rclpy.init(args=args)
    node = ControllerNode(debug=DEBUG, use_gps=USE_GPS)

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
