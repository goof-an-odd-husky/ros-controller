import json
import time
import numpy as np
from scipy.spatial.transform import Rotation
import threading

import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore
from numpy.typing import NDArray

from goof_an_odd_husky.trajectory_planner import TrajectoryPlanner
from goof_an_odd_husky.obstacles import ObstaclePipeline
from goof_an_odd_husky.helpers import gps_to_vector
from goof_an_odd_husky.teb_planner import TEBPlanner
from goof_an_odd_husky.trajectory_visualizer import TrajectoryVisualizer

from goof_an_odd_husky.kinematics import get_odom_delta
from goof_an_odd_husky.trajectory_tracker import trajectory_to_action


class ControllerNode(Node):
    sensor_callback_group: MutuallyExclusiveCallbackGroup
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
    viz_lock: threading.Lock
    pending_trajectory: NDArray[np.float64] | None
    pending_obstacles: list | None
    pending_start_goal: tuple[list[float], list[float]] | None
    visualizer: TrajectoryVisualizer
    planner: TrajectoryPlanner
    obstacle_pipeline: ObstaclePipeline
    initial_start: list[float]
    initial_goal: list[float]
    current_robot_pose_global: list[float]
    goal_reached: bool
    goal_lat_lon: tuple[float, float] | None
    needs_initial_plan: bool
    last_time_ns: int
    use_gps: bool

    def __init__(self, debug: bool = False, use_gps: bool = True) -> None:
        super().__init__("controller_node")
        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

        self.use_gps = use_gps

        self.sensor_callback_group = MutuallyExclusiveCallbackGroup()
        self.control_callback_group = MutuallyExclusiveCallbackGroup()

        self.velocity_publisher = self.create_publisher(
            TwistStamped, "/husky/cmd_vel", 10
        )

        self.lidar_subscription = self.create_subscription(
            LaserScan,
            "/husky/sensors/lidar2d_0/scan",
            self.lidar_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.gps_subscription = None
        if self.use_gps:
            self.gps_subscription = self.create_subscription(
                NavSatFix,
                "/husky/sensors/gps_0/fix",
                self.gps_callback,
                10,
                callback_group=self.sensor_callback_group,
            )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/husky/odometry/global",
            self.odom_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.control_callback_group
        )

        self.data_lock = threading.Lock()
        self.latest_odom = None
        self.latest_scan = None
        self.last_scan_time = None
        self.latest_gps = None
        self.last_processed_odom = None
        self.first_gps = None

        self.viz_lock = threading.Lock()
        self.pending_trajectory = None
        self.pending_obstacles = None
        self.pending_start_goal = None

        self.visualizer = TrajectoryVisualizer(
            x_lim=(-10, 10),
            y_lim=(-4, 16),
            path_render_mode="both",
            interactive_obstacles=False,
            use_gps=use_gps,
            on_goal_set=self.set_new_goal,
        )

        self.initial_start = [0.0, 0.0, 0.0]
        self.initial_goal = [0.0, 0.0, 0.0]
        self.visualizer.set_start_goal(self.initial_start, self.initial_goal)
        self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)

        self.current_robot_pose_global = [0.0, 0.0, 0.0]
        self.goal_reached = False
        self.goal_lat_lon = None
        self.needs_initial_plan = False
        self.last_time_ns = self.get_clock().now().nanoseconds

        self.obstacle_pipeline = ObstaclePipeline(
            cluster_break_distance=2, geometry_split_threshold=2
        )

    def set_new_goal(self, latitude: float, longitude: float) -> None:
        with self.data_lock:
            self.goal_lat_lon = (latitude, longitude)
            self.goal_reached = False
            self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)
            self.needs_initial_plan = True
        self.get_logger().info(
            f"New goal set: {'lat=' if self.use_gps else 'x='}{latitude}, {'lon=' if self.use_gps else 'y='}{longitude}"
        )

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
        lat, lon, alt = (
            self.latest_gps.latitude,
            self.latest_gps.longitude,
            self.latest_gps.altitude,
        )
        self.get_logger().debug(
            f"GPS data received: {lat, lon, alt}", throttle_duration_sec=1.0
        )

    def odom_callback(self, msg: Odometry) -> None:
        with self.data_lock:
            self.latest_odom = msg

    def control_loop(self) -> None:
        start_time = time.perf_counter()
        performance = {}

        with self.data_lock:
            scan, scan_time = self.latest_scan, self.last_scan_time
            odom_g, gps_data = self.latest_odom, self.latest_gps
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
        current_velocity, current_omega = (
            odom_g.twist.twist.linear.x,
            odom_g.twist.twist.angular.z,
        )

        orientation = odom_g.pose.pose.orientation
        yaw = Rotation.from_quat(
            [orientation.x, orientation.y, orientation.z, orientation.w]
        ).as_euler("zyx")[0]

        with self.viz_lock:
            self.current_robot_pose_global = [
                odom_g.pose.pose.position.x,
                odom_g.pose.pose.position.y,
                yaw,
            ]

            if goal_lat_lon is None:
                self.pending_start_goal = ([0.0, 0.0, 0.0], [])
                self.pending_trajectory = np.array([])

        detected_obstacles = self._process_obstacles(scan, scan_time)
        t2 = time.perf_counter()
        performance["Obstacle processing"] = round((t2 - t1) * 1000, 2)

        if detected_obstacles is None:
            self._publish_velocity(0.0, 0.0)
            return

        with self.viz_lock:
            self.pending_obstacles = detected_obstacles

        if goal_lat_lon is None:
            self.get_logger().debug("Waiting for goal...", throttle_duration_sec=2.0)
            performance["Total"] = round((time.perf_counter() - start_time) * 1000, 2)
            self.get_logger().debug(
                "Performance:\n" + json.dumps(performance, indent=4),
                throttle_duration_sec=2.0,
            )
            self._publish_velocity(0.0, 0.0)
            return

        pending_start_goal = self._update_local_goal(goal_lat_lon, odom_g)
        t3 = time.perf_counter()
        performance["Goal update"] = round((t3 - t2) * 1000, 2)

        if pending_start_goal is None:
            self._publish_velocity(0.0, 0.0)
            return

        with self.viz_lock:
            self.pending_start_goal = pending_start_goal

        if self.planner.get_distance_goal() < 1.0:
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info("Goal Reached")
                self._publish_velocity(0.0, 0.0)
                self.goal_lat_lon = None

                with self.viz_lock:
                    self.pending_start_goal = ([0.0, 0.0, 0.0], [])
                    self.pending_trajectory = np.array([])
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
            x1, y1 = odom_g.pose.pose.position.x, odom_g.pose.pose.position.y
            yaw1 = yaw

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

                with self.viz_lock:
                    self.current_robot_pose_global = [x2, y2, yaw2]

        v, omega = trajectory_to_action(trajectory)
        self._publish_velocity(v, omega)

        t5 = time.perf_counter()
        performance["Publish/Finish"] = round((t5 - t4) * 1000, 2)
        performance["Total"] = round((t5 - start_time) * 1000, 2)
        self.get_logger().debug(
            "Performance:\n" + json.dumps(performance, indent=4),
            throttle_duration_sec=2.0,
        )

        with self.viz_lock:
            self.pending_trajectory, self.pending_start_goal = (
                trajectory,
                pending_start_goal,
            )

    def _update_odometry(self, odom_g) -> None:
        dx, dy, dt = get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

    def _update_local_goal(self, goal_lat_lon, odom_g) -> tuple | None:
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

        vehicle_x, vehicle_y = odom_g.pose.pose.position.x, odom_g.pose.pose.position.y
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

        return (self.initial_start, [local_x, local_y, 0.0])

    def _process_obstacles(self, scan, scan_time) -> list | None:
        if (self.get_clock().now() - scan_time).nanoseconds / 1e9 > 0.5:
            self.get_logger().error("Lidar data stale", throttle_duration_sec=1.0)
            return None
        obstacles = self.obstacle_pipeline.process(scan)
        self.planner.update_obstacles(obstacles)
        return obstacles

    def _publish_velocity(self, v: float, omega: float) -> None:
        twist_stamped_message = TwistStamped()
        twist_stamped_message.twist.linear.x = float(v)
        twist_stamped_message.twist.angular.z = float(omega)
        self.velocity_publisher.publish(twist_stamped_message)

    def render_loop(self) -> None:
        if not self.visualizer.is_open:
            return

        with self.viz_lock:
            trajectory = self.pending_trajectory
            obstacles = self.pending_obstacles
            start_goal = self.pending_start_goal
            robot_pose = self.current_robot_pose_global

        if trajectory is None and obstacles is None and start_goal is None:
            return

        self.visualizer.update_world_state(
            robot_pose=robot_pose,
            trajectory=trajectory,
            obstacles=obstacles,
            start_goal=start_goal,
        )
        self.visualizer.draw()


def main(args=None) -> None:
    rclpy.init(args=args)

    controller_node = ControllerNode(debug=True, use_gps=False)

    executor = MultiThreadedExecutor(num_threads=4)
    executor.add_node(controller_node)

    spin_thread = threading.Thread(target=executor.spin, daemon=True)
    spin_thread.start()

    qt_timer = QtCore.QTimer()
    qt_timer.timeout.connect(controller_node.render_loop)
    qt_timer.start(100)

    try:
        pg.exec()
    except KeyboardInterrupt:
        pass
    finally:
        controller_node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
