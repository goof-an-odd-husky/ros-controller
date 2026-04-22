import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Empty, Header

from goof_an_odd_husky_common.config import (
    TOPICS,
    USE_GPS,
    DEBUG,
    HEARTBEAT_TIMEOUT_SEC,
)
from goof_an_odd_husky_common.qos import LATCHED_QOS
from goof_an_odd_husky_common.types import GpsCoord

from goof_an_odd_husky.performance_tracker import PerformanceTracker
from goof_an_odd_husky.sensor_cache import SensorCache
from goof_an_odd_husky.orchestrator import NavigationOrchestrator
from goof_an_odd_husky.publisher_communicator import PublisherCommunicator


class ControllerNode(Node):
    """Main ROS 2 Node orchestrating global and local navigation.

    Attributes:
        use_gps: Whether to use GPS for navigation.
        sensor_cache: Thread-safe data storage for subscription data.
        orchestrator: The logic engine for processing navigation steps.
        publisher: Communicator for handling all ROS 2 publications.
    """

    use_gps: bool
    sensor_cache: SensorCache
    orchestrator: NavigationOrchestrator
    publisher: PublisherCommunicator

    def __init__(self, use_gps: bool, debug: bool = False) -> None:
        """Initialize the ControllerNode.

        Args:
            use_gps: Whether to use GPS for navigation.
            debug: Whether to set the logger level to DEBUG.
        """
        super().__init__("controller_node")
        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.use_gps = use_gps

        self.sensor_cache = SensorCache()
        self.orchestrator = NavigationOrchestrator(
            use_gps=self.use_gps, logger=self.get_logger()
        )
        self.publisher = PublisherCommunicator(self)

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
            Header,
            "/nav/cancel",
            self.cancel_callback,
            LATCHED_QOS,
            callback_group=self.nav_cmd_cb_group,
        )
        self.heartbeat_subscription = self.create_subscription(
            Empty,
            "/viz/heartbeat",
            self.heartbeat_callback,
            10,
            callback_group=self.nav_cmd_cb_group,
        )

        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.control_callback_group
        )

        self.publisher.publish_status("idle")

    def heartbeat_callback(self, _: Empty) -> None:
        """Callback for the visualizer heartbeat to track UI connection status."""
        self.sensor_cache.update_heartbeat(self.get_clock().now())

    def goal_callback(self, msg: PoseStamped) -> None:
        """Callback for receiving a new global goal."""
        self.last_goal_stamp = msg.header.stamp
        self.orchestrator.set_goal(GpsCoord(msg.pose.position.x, msg.pose.position.y))

        self.publisher.publish_status("navigating")
        self.get_logger().info(
            f"New goal set: {msg.pose.position.x}, {msg.pose.position.y}"
        )

    def cancel_callback(self, msg: Header) -> None:
        """Callback for cancelling the current navigation goal."""
        last_goal = getattr(self, "last_goal_stamp", None)
        cancel_is_newer = last_goal is None or rclpy.time.Time.from_msg(
            msg.stamp
        ) > rclpy.time.Time.from_msg(last_goal)

        if not cancel_is_newer:
            return

        self.orchestrator.cancel_goal()

        self.publisher.publish_velocity(0.0, 0.0, self.get_clock().now().to_msg())
        self.publisher.publish_status("idle")
        self.get_logger().info("Navigation cancelled")

    def lidar_callback(self, msg: LaserScan) -> None:
        """Callback for LiDAR scan updates."""
        self.sensor_cache.update_scan(msg, self.get_clock().now())

    def gps_callback(self, msg: NavSatFix) -> None:
        """Callback for GPS position updates."""
        gps_coord = GpsCoord(msg.latitude, msg.longitude)
        self.sensor_cache.update_gps(gps_coord, msg.status.status >= 0)

    def odom_callback(self, msg: Odometry) -> None:
        """Callback for global Odometry updates."""
        self.sensor_cache.update_odom(msg)

    def control_loop(self) -> None:
        """Main control loop responsible for executing planner step and publishing commands."""
        with PerformanceTracker(
            self.get_logger(), log_level="debug", throttle_sec=2.0
        ) as performance:
            odom, scan, scan_time, gps = self.sensor_cache.get_core_sensors()
            performance.update("Data acquisition")

            scan_age_sec = None
            current_time = self.get_clock().now()
            if scan_time is not None:
                scan_age_sec = (current_time - scan_time).nanoseconds / 1e9

            out = self.orchestrator.step(
                odom_g=odom,
                scan=scan,
                scan_age_sec=scan_age_sec,
                gps_data=gps,
                first_gps=self.sensor_cache.get_first_gps(),
                current_time_sec=current_time.nanoseconds / 1e9,
                visualizer_alive=self.sensor_cache.is_visualizer_alive(
                    self.get_clock().now(), HEARTBEAT_TIMEOUT_SEC
                ),
                get_latest_odom=self.sensor_cache.get_odom,
                performance=performance,
            )

            performance.update("Publish/Finish")

            self.publisher.publish_robot_pose(
                out.robot_pose
            ) if out.robot_pose else None
            self.publisher.publish_status(out.status) if out.status else None
            self.publisher.publish_global_path(out.global_path)
            self.publisher.publish_obstacles(out.obstacles)
            self.publisher.publish_trajectory(out.trajectory)
            self.publisher.publish_goal_local(out.goal_local)

            if out.v is not None and out.omega is not None:
                self.publisher.publish_velocity(
                    out.v, out.omega, self.get_clock().now().to_msg()
                )


def main(args=None) -> None:
    """Entry point for running the controller node."""
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
