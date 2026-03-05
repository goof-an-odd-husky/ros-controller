import json
import time
from goof_an_odd_husky.trajectory_planner import (
    TrajectoryPlanner,
    ObstaclePipeline,
    CircleObstacle,
    LineObstacle,
)
from numpy.typing import NDArray
from typing import Annotated
from goof_an_odd_husky.helpers import gps_to_vector, normalize_angle
import rclpy
from rclpy.node import Node
from rclpy.executors import MultiThreadedExecutor
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from sensor_msgs.msg import LaserScan, NavSatFix
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TwistStamped
import numpy as np
from scipy.spatial.transform import Rotation
import threading

import pyqtgraph as pg
from pyqtgraph.Qt import QtCore

from goof_an_odd_husky.teb_planner import TEBPlanner
from goof_an_odd_husky.trajectory_visualizer import TrajectoryVisualizer


class ControllerNode(Node):
    planner: TrajectoryPlanner
    obstacle_pipeline: ObstaclePipeline
    visualizer: TrajectoryVisualizer
    velocity_publisher: rclpy.publisher.Publisher
    lidar_subscription: rclpy.subscription.Subscription
    gps_subscription: rclpy.subscription.Subscription
    odom_subscription: rclpy.subscription.Subscription
    control_timer: rclpy.timer.Timer
    sensor_callback_group: MutuallyExclusiveCallbackGroup
    control_callback_group: MutuallyExclusiveCallbackGroup
    latest_odom: Odometry | None
    last_processed_odom: Odometry | None
    latest_scan: LaserScan | None
    last_scan_time: rclpy.time.Time | None
    latest_gps: NavSatFix | None
    first_gps: NavSatFix | None
    first_yaw: float | None
    data_lock: threading.Lock
    pending_trajectory: NDArray[np.float64] | None
    pending_obstacles: NDArray[np.float64] | None
    pending_start_goal: tuple[list[float | None, list[float]]]
    viz_lock: threading.Lock
    initial_start: list[float]
    initial_goal: list[float]
    current_robot_pose_global: list[float]
    last_time_ns: int
    goal_reached: bool
    goal_lat_lon: tuple[float, float] | None

    def __init__(self, debug: bool = False) -> None:
        super().__init__("controller_node")
        if debug:
            self.get_logger().set_level(rclpy.logging.LoggingSeverity.DEBUG)

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

        self.gps_subscription = self.create_subscription(
            NavSatFix,
            "/husky/sensors/gps_0/fix",
            self.gps_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.odom_subscription = self.create_subscription(
            Odometry,
            "/husky/platform/odom/filtered",
            self.odom_callback,
            10,
            callback_group=self.sensor_callback_group,
        )

        self.control_timer = self.create_timer(
            0.1, self.control_loop, callback_group=self.control_callback_group
        )

        self.latest_odom = None
        self.last_processed_odom = None

        self.latest_scan = None
        self.last_scan_time = None
        self.latest_gps = None
        self.first_gps = None
        self.first_yaw = None
        self.data_lock = threading.Lock()

        self.pending_trajectory = None
        self.pending_obstacles = None
        self.pending_start_goal = None
        self.viz_lock = threading.Lock()

        self.visualizer = TrajectoryVisualizer(
            x_lim=(-10, 10),
            y_lim=(-4, 16),
            path_render_mode="both",
            interactive_obstacles=False,
            on_goal_set=self.set_new_goal,
        )

        self.initial_start = [0.0, 0.0, 0.0]
        self.initial_goal = [10.0, 0.0, 0.0]
        self.visualizer.set_start_goal(self.initial_start, self.initial_goal)
        self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)
        self.planner.plan()

        self.current_robot_pose_global = [0.0, 0.0, 0.0]
        self.goal_reached = False
        self.goal_lat_lon = None

        self.obstacle_pipeline = ObstaclePipeline()

        self.get_logger().info("Controller initialized with Planner and Visualizer")
        self.last_time_ns = self.get_clock().now().nanoseconds

    def set_new_goal(self, latitude: float, longitude: float) -> None:
        with self.data_lock:
            self.goal_lat_lon = (latitude, longitude)
            self.goal_reached = False
            self.last_processed_odom = None

            self.planner = TEBPlanner(self.initial_start, self.initial_goal, 1, 2)
            self.planner.plan()

        self.get_logger().info(f"New goal set: lat={latitude}, lon={longitude}")

    def lidar_callback(self, msg: LaserScan) -> None:
        with self.data_lock:
            self.latest_scan = msg
            self.last_scan_time = self.get_clock().now()
        self.get_logger().debug("Lidar data received", throttle_duration_sec=5.0)

    def gps_callback(self, msg: NavSatFix) -> None:
        with self.data_lock:
            self.latest_gps = msg
            if not self.first_gps:
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
            if self.first_yaw is None:
                self.first_yaw = Rotation.from_quat(
                    [
                        msg.pose.pose.orientation.x,
                        msg.pose.pose.orientation.y,
                        msg.pose.pose.orientation.z,
                        msg.pose.pose.orientation.w,
                    ]
                ).as_euler("zyx")[0]

    def get_odom_delta(
        self, curr: Odometry, prev: Odometry | None
    ) -> tuple[float, float, float]:
        if prev is None:
            return 0.0, 0.0, 0.0

        curr_pos = curr.pose.pose.position
        prev_pos = prev.pose.pose.position

        curr_yaw = Rotation.from_quat(
            [
                curr.pose.pose.orientation.x,
                curr.pose.pose.orientation.y,
                curr.pose.pose.orientation.z,
                curr.pose.pose.orientation.w,
            ]
        ).as_euler("zyx")[0]
        prev_yaw = Rotation.from_quat(
            [
                prev.pose.pose.orientation.x,
                prev.pose.pose.orientation.y,
                prev.pose.pose.orientation.z,
                prev.pose.pose.orientation.w,
            ]
        ).as_euler("zyx")[0]

        dx_global = curr_pos.x - prev_pos.x
        dy_global = curr_pos.y - prev_pos.y
        dtheta = normalize_angle(curr_yaw - prev_yaw)

        c = np.cos(prev_yaw)
        s = np.sin(prev_yaw)

        dx_local = dx_global * c + dy_global * s
        dy_local = -dx_global * s + dy_global * c

        return dx_local, dy_local, dtheta

    def trajectory_to_action(
        self, trajectory: Annotated[NDArray[np.float64], (None, 4)]
    ) -> tuple[float, float]:
        if trajectory is None or len(trajectory) < 2:
            return 0.0, 0.0

        x1, y1, th1, dt = trajectory[0]
        x2, y2, th2, _ = trajectory[1]

        if dt < 1e-3:
            return 0.0, 0.0

        dx = x2 - x1
        dy = y2 - y1

        c, s = np.cos(th1), np.sin(th1)
        direction = 1.0
        if (dx * c + dy * s) < 0:
            direction = -1.0

        chord = np.hypot(dx, dy)
        arc_angle = normalize_angle(th2 - th1)
        if abs(arc_angle) < 1e-5:
            arc_length = chord
        else:
            circle_radius = chord / (2 * np.sin(arc_angle / 2))
            arc_length = circle_radius * arc_angle

        v = direction * abs(arc_length) / dt
        omega = arc_angle / dt

        return v, omega

    def control_loop(self) -> None:
        loop_start_time = time.perf_counter()
        performance = {}

        with self.data_lock:
            scan = self.latest_scan
            scan_time = self.last_scan_time
            gps_data = self.latest_gps
            odom = self.latest_odom
            goal_lat_lon = self.goal_lat_lon

        t1 = time.perf_counter()
        performance["Data acquision"] = round((t1 - loop_start_time) * 1000, 2)

        if scan is None:
            self.get_logger().warn("Waiting for LIDAR...", throttle_duration_sec=2.0)
            return
        if odom is None:
            self.get_logger().warn("Waiting for Odometry...", throttle_duration_sec=2.0)
            return
        if gps_data is None:
            self.get_logger().warn("Waiting for GPS...", throttle_duration_sec=2.0)
            return

        self._update_odometry(odom)

        detected_obstacles = self._process_obstacles(scan, scan_time)
        t2 = time.perf_counter()
        performance["Obstacle processing"] = round((t2 - t1) * 1000, 2)

        if detected_obstacles is None:
            self.get_logger().warn("Error processing obstacles")
            self._publish_velocity(0.0, 0.0)
            return

        with self.viz_lock:
            self.pending_obstacles = detected_obstacles

        if goal_lat_lon is None:
            self.get_logger().debug(
                "Waiting for goal coordinates from UI...", throttle_duration_sec=2.0
            )
            self.get_logger().debug(
                "Performance:\n" + json.dumps(performance, indent=4),
                throttle_duration_sec=2.0,
            )
            self._publish_velocity(0.0, 0.0)
            return

        pending_start_goal = self._update_local_goal(gps_data, goal_lat_lon, odom)
        t3 = time.perf_counter()
        performance["Goal update"] = round((t3 - t2) * 1000, 2)

        if self.planner.get_length() <= 2:
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info("Reached the goal - stopping control loop")
                self._publish_velocity(0.0, 0.0)
                self.goal_lat_lon = None
            return

        self.goal_reached = False

        self.planner.refine()
        t4 = time.perf_counter()
        performance["Planner refinement"] = round((t4 - t3) * 1000, 2)

        trajectory = self.planner.get_trajectory()

        v, omega = self.trajectory_to_action(trajectory)
        self.get_logger().debug(f"{v=}, {omega=}", throttle_duration_sec=1.0)
        self._publish_velocity(v, omega)

        self.last_time_ns = self.get_clock().now().nanoseconds

        t5 = time.perf_counter()
        performance["Publish/Finish"] = round((t5 - t4) * 1000, 2)
        performance["Total"] = round((t5 - loop_start_time) * 1000, 2)
        self.get_logger().debug(
            "Performance:\n" + json.dumps(performance, indent=4),
            throttle_duration_sec=2.0,
        )

        with self.viz_lock:
            self.pending_trajectory = trajectory
            if pending_start_goal is not None:
                self.pending_start_goal = pending_start_goal

    def _update_odometry(self, odom) -> None:
        """Updates the planner's trajectory based on odometry deltas."""
        dx, dy, dtheta = self.get_odom_delta(odom, self.last_processed_odom)

        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dtheta)

        self.last_processed_odom = odom

    def _process_obstacles(self, scan, scan_time) -> list | None:
        """Checks Lidar freshness, processes scan, and updates planner. Returns None if stale."""
        scan_age = (self.get_clock().now() - scan_time).nanoseconds / 1e9
        if scan_age > 0.5:
            self.get_logger().error(
                "Lidar data stale, stopping", throttle_duration_sec=1.0
            )
            return None

        detected_obstacles = self.obstacle_pipeline.process(scan)
        self.planner.update_obstacles(detected_obstacles)
        return detected_obstacles

    def _update_local_goal(self, gps_data, goal_lat_lon, odom) -> tuple | None:
        """Converts global GPS coordinates to the local frame and moves the planner goal."""
        if gps_data.status.status < 0:
            self.get_logger().warn(
                "GPS received but no fix acquired", throttle_duration_sec=2.0
            )
            return None

        displacement = gps_to_vector(
            self.first_gps.latitude,
            self.first_gps.longitude,
            gps_data.latitude,
            gps_data.longitude,
        )

        goal_displacement = gps_to_vector(
            self.first_gps.latitude,
            self.first_gps.longitude,
            goal_lat_lon[0],
            goal_lat_lon[1],
        )

        c_init = np.cos(self.first_yaw)
        s_init = np.sin(self.first_yaw)

        disp_forward = displacement[0] * c_init + displacement[1] * s_init
        disp_left = -displacement[0] * s_init + displacement[1] * c_init

        goal_forward = goal_displacement[0] * c_init + goal_displacement[1] * s_init
        goal_left = -goal_displacement[0] * s_init + goal_displacement[1] * c_init

        goal_x_initial_frame = goal_forward - disp_forward
        goal_y_initial_frame = goal_left - disp_left

        current_yaw = Rotation.from_quat(
            [
                odom.pose.pose.orientation.x,
                odom.pose.pose.orientation.y,
                odom.pose.pose.orientation.z,
                odom.pose.pose.orientation.w,
            ]
        ).as_euler("zyx")[0]

        if self.first_yaw is not None:
            current_yaw = normalize_angle(current_yaw - self.first_yaw)

        c = np.cos(-current_yaw)
        s = np.sin(-current_yaw)

        local_goal_x = goal_x_initial_frame * c - goal_y_initial_frame * s
        local_goal_y = goal_x_initial_frame * s + goal_y_initial_frame * c

        local_goal = [local_goal_x, local_goal_y, 0.0]
        self.planner.move_goal([local_goal_x, local_goal_y], [0.0], 0.5, 5.0)

        with self.viz_lock:
            self.current_robot_pose_global = [
                displacement[0],
                displacement[1],
                current_yaw,
            ]

        return (self.initial_start, local_goal)

    def _publish_velocity(self, v: float, omega: float) -> None:
        """Helper method to construct and publish a TwistStamped message."""
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
            rx, ry, rtheta = self.current_robot_pose_global

            if self.first_yaw is not None:
                rtheta = normalize_angle(rtheta - self.first_yaw)

            self.pending_trajectory = None
            self.pending_obstacles = None
            self.pending_start_goal = None

        if trajectory is None and obstacles is None:
            return

        if self.visualizer.use_global:
            c = np.cos(rtheta)
            s = np.sin(rtheta)

            if trajectory is not None:
                gtraj = trajectory.copy()
                gtraj[:, 0] = rx + trajectory[:, 0] * c - trajectory[:, 1] * s
                gtraj[:, 1] = ry + trajectory[:, 0] * s + trajectory[:, 1] * c
                gtraj[:, 2] = trajectory[:, 2] + rtheta
                trajectory = gtraj

            if obstacles is not None:
                gobs = []
                for obs in obstacles:
                    if isinstance(obs, CircleObstacle):
                        gx = rx + obs.x * c - obs.y * s
                        gy = ry + obs.x * s + obs.y * c
                        gobs.append(CircleObstacle(gx, gy, obs.radius))
                    elif isinstance(obs, LineObstacle):
                        gx1 = rx + obs.x1 * c - obs.y1 * s
                        gy1 = ry + obs.x1 * s + obs.y1 * c
                        gx2 = rx + obs.x2 * c - obs.y2 * s
                        gy2 = ry + obs.x2 * s + obs.y2 * c
                        gobs.append(LineObstacle(gx1, gy1, gx2, gy2))
                obstacles = gobs

            if start_goal is not None:
                start, goal = start_goal
                gsx = rx + start[0] * c - start[1] * s
                gsy = ry + start[0] * s + start[1] * c
                ggx = rx + goal[0] * c - goal[1] * s
                ggy = ry + goal[0] * s + goal[1] * c

                gs_theta = (start[2] if len(start) > 2 else 0.0) + rtheta
                gg_theta = (goal[2] if len(goal) > 2 else 0.0) + rtheta

                start_goal = ([gsx, gsy, gs_theta], [ggx, ggy, gg_theta])

        if start_goal is not None:
            self.visualizer.set_start_goal(*start_goal)
        if obstacles is not None:
            self.visualizer.set_obstacles(obstacles)
        if trajectory is not None:
            self.visualizer.update_trajectory(trajectory)

        self.visualizer.draw()


def main(args=None) -> None:
    rclpy.init(args=args)

    controller_node = ControllerNode(debug=False)

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
