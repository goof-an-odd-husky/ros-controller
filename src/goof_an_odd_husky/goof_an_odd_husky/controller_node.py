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
    sensor_callback_group: MutuallyExclusiveCallbackGroup
    control_callback_group: MutuallyExclusiveCallbackGroup
    velocity_publisher: rclpy.publisher.Publisher
    lidar_subscription: rclpy.subscription.Subscription
    gps_subscription: rclpy.subscription.Subscription
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
        self.get_logger().info(f"New goal set: lat={latitude}, lon={longitude}")

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

    def get_odom_delta(
        self, curr: Odometry, prev: Odometry | None
    ) -> tuple[float, float, float]:
        if prev is None:
            return 0.0, 0.0, 0.0
        curr_pos, prev_pos = curr.pose.pose.position, prev.pose.pose.position
        curr_q, prev_q = curr.pose.pose.orientation, prev.pose.pose.orientation
        curr_yaw = Rotation.from_quat(
            [curr_q.x, curr_q.y, curr_q.z, curr_q.w]
        ).as_euler("zyx")[0]
        prev_yaw = Rotation.from_quat(
            [prev_q.x, prev_q.y, prev_q.z, prev_q.w]
        ).as_euler("zyx")[0]
        dx_global, dy_global = curr_pos.x - prev_pos.x, curr_pos.y - prev_pos.y
        dtheta = normalize_angle(curr_yaw - prev_yaw)
        cos_p, sin_p = np.cos(prev_yaw), np.sin(prev_yaw)
        dx_local = dx_global * cos_p + dy_global * sin_p
        dy_local = -dx_global * sin_p + dy_global * cos_p
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
        dx, dy = x2 - x1, y2 - y1
        direction = 1.0 if (dx * np.cos(th1) + dy * np.sin(th1)) >= 0 else -1.0
        chord = np.hypot(dx, dy)
        arc_angle = normalize_angle(th2 - th1)
        arc_len = (
            chord
            if abs(arc_angle) < 1e-5
            else (chord / (2 * np.sin(arc_angle / 2))) * arc_angle
        )
        return (direction * abs(arc_len) / dt), (arc_angle / dt)

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
        if gps_data is None:
            self.get_logger().warn("Waiting for GPS...", throttle_duration_sec=2.0)
            return

        self._update_odometry(odom_g)
        current_velocity, current_omega = (
            odom_g.twist.twist.linear.x,
            odom_g.twist.twist.angular.z,
        )

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

        if self.planner.get_distance_goal() < 1.0:
            if not self.goal_reached:
                self.goal_reached = True
                self.get_logger().info("Goal Reached")
                self._publish_velocity(0.0, 0.0)
                self.goal_lat_lon = None
            return

        self.goal_reached = False
        self.planner.refine(
            current_velocity=current_velocity, current_omega=current_omega
        )
        t4 = time.perf_counter()
        performance["Planner refinement"] = round((t4 - t3) * 1000, 2)

        trajectory = self.planner.get_trajectory()
        v, omega = self.trajectory_to_action(trajectory)
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
        dx, dy, dt = self.get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

    def _update_local_goal(self, goal_lat_lon, odom_g) -> tuple | None:
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
        with self.viz_lock:
            self.current_robot_pose_global = [vehicle_x, vehicle_y, yaw]
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
            trajectory, obstacles, start_goal = (
                self.pending_trajectory,
                self.pending_obstacles,
                self.pending_start_goal,
            )
            rx, ry, rtheta = self.current_robot_pose_global
            self.pending_trajectory = self.pending_obstacles = (
                self.pending_start_goal
            ) = None
        if trajectory is None and obstacles is None:
            return
        if self.visualizer.use_global:
            c, s = np.cos(rtheta), np.sin(rtheta)
            if trajectory is not None:
                gtraj = trajectory.copy()
                gtraj[:, 0], gtraj[:, 1] = (
                    rx + trajectory[:, 0] * c - trajectory[:, 1] * s,
                    ry + trajectory[:, 0] * s + trajectory[:, 1] * c,
                )
                gtraj[:, 2] = trajectory[:, 2] + rtheta
                trajectory = gtraj
            if obstacles is not None:
                graphical_obs = []
                for o in obstacles:
                    if isinstance(o, CircleObstacle):
                        graphical_obs.append(
                            CircleObstacle(
                                rx + o.x * c - o.y * s, ry + o.x * s + o.y * c, o.radius
                            )
                        )
                    elif isinstance(o, LineObstacle):
                        graphical_obs.append(
                            LineObstacle(
                                rx + o.x1 * c - o.y1 * s,
                                ry + o.x1 * s + o.y1 * c,
                                rx + o.x2 * c - o.y2 * s,
                                ry + o.x2 * s + o.y2 * c,
                            )
                        )
                obstacles = graphical_obs
            if start_goal is not None:
                st, gl = start_goal
                gsx, gsy = rx + st[0] * c - st[1] * s, ry + st[0] * s + st[1] * c
                ggx, ggy = rx + gl[0] * c - gl[1] * s, ry + gl[0] * s + gl[1] * c
                start_goal = ([gsx, gsy, st[2] + rtheta], [ggx, ggy, gl[2] + rtheta])
        if start_goal:
            self.visualizer.set_start_goal(*start_goal)
        if obstacles:
            self.visualizer.set_obstacles(obstacles)
        if trajectory is not None:
            self.visualizer.update_trajectory(trajectory)
        self.visualizer.draw()


def main(args=None) -> None:
    rclpy.init(args=args)

    controller_node = ControllerNode(debug=True)

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
