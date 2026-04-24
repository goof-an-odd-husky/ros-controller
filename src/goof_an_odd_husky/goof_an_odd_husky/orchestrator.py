import math
from dataclasses import dataclass
from typing import Callable

from rclpy.impl.rcutils_logger import RcutilsLogger
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import Odometry

from goof_an_odd_husky_common.config import (
    MAX_TRAJECTORY_DISTANCE,
    SAFETY_RADIUS,
    BARRIER_OFFSET,
    SIM,
    MAX_V,
    MAX_OMEGA,
    MAX_A,
    INITIAL_STEP,
    CLUSTER_BREAK_DISTANCE,
    GEOMETRY_SPLIT_THRESHOLD,
    MIN_SCAN_RANGE,
    MEDIAN_FILTER_SIZE,
    SOFTMIN_ALPHA,
    TRAJECTORY_LIMITS,
    TEB_WEIGHTS,
    MAX_TEB_ITERATIONS,
    OSM_RELATION_ID,
    MIN_CIRCLE_RADIUS,
    MAX_CIRCLE_RADIUS,
    MAX_LINE_DISTANCE,
    CRITICAL_STOP_RADIUS,
    MAX_PITCH,
    MAX_ROLL,
    STUCK_TIMEOUT,
    STUCK_VEL_TOLERANCE,
    MAX_CROSS_TRACK_ERROR,
    TRAJECTORY_COLLISION_LOOKAHEAD,
    MAX_ODOM_JUMP_SPEED_LINEAR,
    MAX_ODOM_JUMP_SPEED_ANGULAR,
    MAX_ESTOP_V,
    MAX_ESTOP_OMEGA,
)
from goof_an_odd_husky_common.helpers import quat_to_yaw, quat_to_euler
from goof_an_odd_husky_common.types import Pose2D, Trajectory, GpsCoord
from goof_an_odd_husky_common.obstacles import Obstacle

from goof_an_odd_husky.performance_tracker import PerformanceTracker
from goof_an_odd_husky.local_navigation.teb_planner import TEBPlanner
from goof_an_odd_husky.local_navigation.obstacle_pipeline import ObstaclePipeline
from goof_an_odd_husky.local_navigation.kinematics import get_odom_delta
from goof_an_odd_husky.local_navigation.local_goal_selector import LocalGoalSelector
from goof_an_odd_husky.global_navigation.path_manager import GlobalPathManager
from goof_an_odd_husky.action import trajectory_to_action
from goof_an_odd_husky.local_navigation.safety import is_segment_safe
from goof_an_odd_husky.local_navigation.astar_local import astar_visibility_path


@dataclass
class OrchestratorOutput:
    """Contains logic-computed states meant to be pushed directly out to publishers.

    Attributes:
        v: Linear velocity.
        omega: Angular velocity.
        status: Status string to publish.
        robot_pose: Current pose for visualization.
        trajectory: Planned trajectory for visualization.
        global_path: Current global path snapshot for visualization.
        obstacles: Detected obstacles for visualization.
        goal_local: Immediate target point for visualization.
    """

    v: float | None = None
    omega: float | None = None
    status: str | None = None
    robot_pose: Pose2D | None = None
    trajectory: Trajectory | None = None
    global_path: list[tuple[float, float]] | None = None
    obstacles: list[Obstacle] | None = None
    goal_local: list[float] | None = None


class NavigationOrchestrator:
    """Core algorithmic logic for navigation, completely decoupled from ROS 2 execution mechanics.

    Attributes:
        use_gps: Boolean indicating whether to use GPS for global localization.
        logger: ROS 2 logger for reporting status.
        initial_start: Starting pose for the TEB planner.
        initial_goal: Initial goal pose for the TEB planner.
        last_processed_odom: The odometry message used in the last planner loop.
        planner: The TEB local planner instance.
        needs_initial_plan: Boolean flag to trigger an initial TEB plan.
        needs_astar_plan: Boolean flag to trigger an A* plan.
        path_manager: Manager for the global A* path logic.
        goal_selector: Selector for the immediate local goal.
        obstacle_pipeline: Pipeline for processing LiDAR scans into geometric obstacles.
        last_cmd_v: Linear velocity commanded in the previous iteration.
        last_cmd_omega: Angular velocity commanded in the previous iteration.
        mismatch_start_time: Timestamp when velocity mismatch was first detected.
    """

    use_gps: bool
    logger: RcutilsLogger
    initial_start: list[float]
    initial_goal: list[float]
    last_processed_odom: Odometry | None
    planner: TEBPlanner
    needs_initial_plan: bool
    needs_astar_plan: bool
    path_manager: GlobalPathManager
    goal_selector: LocalGoalSelector
    obstacle_pipeline: ObstaclePipeline
    last_cmd_v: float
    last_cmd_omega: float
    mismatch_start_time: float | None

    def __init__(self, use_gps: bool, logger: RcutilsLogger) -> None:
        """Initialize the NavigationOrchestrator.

        Args:
            use_gps: Whether to use GPS for navigation.
            logger: ROS 2 logger instance.
        """
        self.use_gps = use_gps
        self.logger = logger

        self.initial_start = [0.0, 0.0, 0.0]
        self.initial_goal = [0.0, 0.0, 0.0]
        self.last_processed_odom = None

        self.planner = TEBPlanner(
            self.initial_start,
            self.initial_goal,
            max_v=MAX_V,
            max_omega=MAX_OMEGA,
            max_a=MAX_A,
            initial_step=INITIAL_STEP,
            safety_radius=SAFETY_RADIUS,
            softmin_alpha=SOFTMIN_ALPHA,
            trajectory_limits=TRAJECTORY_LIMITS,
            weights=TEB_WEIGHTS,
        )
        self.needs_initial_plan = False
        self.needs_astar_plan = False

        self.path_manager = GlobalPathManager(
            use_gps=self.use_gps,
            logger=self.logger,
            osm_relation_id=OSM_RELATION_ID,
        )
        self.goal_selector = LocalGoalSelector(
            max_trajectory_distance=MAX_TRAJECTORY_DISTANCE,
            logger=self.logger,
            safety_radius=SAFETY_RADIUS,
        )
        self.obstacle_pipeline = ObstaclePipeline(
            CLUSTER_BREAK_DISTANCE,
            GEOMETRY_SPLIT_THRESHOLD,
            MIN_SCAN_RANGE,
            MEDIAN_FILTER_SIZE,
            MIN_CIRCLE_RADIUS,
            MAX_CIRCLE_RADIUS,
            MAX_LINE_DISTANCE,
        )

        self.last_cmd_v = 0.0
        self.last_cmd_omega = 0.0
        self.mismatch_start_time = None

    def set_goal(self, gps_coord: GpsCoord) -> None:
        """Configures the sub-systems for a newly requested global goal.

        Args:
            gps_coord: The coordinate of the new goal.
        """
        self.path_manager.set_goal(gps_coord)
        self.planner = TEBPlanner(
            self.initial_start,
            self.initial_goal,
            max_v=MAX_V,
            max_omega=MAX_OMEGA,
            max_a=MAX_A,
            initial_step=INITIAL_STEP,
            safety_radius=SAFETY_RADIUS,
            softmin_alpha=SOFTMIN_ALPHA,
            trajectory_limits=TRAJECTORY_LIMITS,
            weights=TEB_WEIGHTS,
        )
        self.needs_initial_plan = True

    def cancel_goal(self) -> None:
        """Clears current paths and stops planning execution."""
        self.path_manager.cancel_goal()
        self.needs_initial_plan = False

    def _log_missing_sensors(
        self, scan: LaserScan | None, odom: Odometry | None, gps: GpsCoord | None
    ) -> None:
        """Helper to log warnings when required sensors are absent.

        Args:
            scan: Cached Lidar scan.
            odom: Cached odometry.
            gps: Cached GPS coordinate.
        """
        missing = []
        if scan is None:
            missing.append("Lidar")
        if odom is None:
            missing.append("Odom")
        if self.use_gps and gps is None:
            missing.append("GPS")
        self.logger.warn(
            f"Waiting for sensors ({', '.join(missing)})...", throttle_duration_sec=2.0
        )

    def _log_robot_location(
        self, robot_pose: Pose2D, gps_data: GpsCoord | None
    ) -> None:
        """Helper to log the current robot location.

        Args:
            robot_pose: Current 2D pose.
            gps_data: Current GPS coordinate.
        """
        if self.use_gps and gps_data:
            msg = (
                f"GPS: Lat={gps_data.lat:.6f}, "
                f"Lon={gps_data.lon:.6f}, "
                f"Yaw={robot_pose.theta:.2f}rad"
            )
        else:
            msg = f"Pose: X={robot_pose.x:.2f}m, Y={robot_pose.y:.2f}m, Yaw={robot_pose.theta:.2f}rad"
        self.logger.info(msg, throttle_duration_sec=2.0)

    def _trigger_estop(
        self, out: OrchestratorOutput, reason: str
    ) -> OrchestratorOutput:
        """Clears velocities and sets status to E-Stop.

        Args:
            out: Orchestrator output container.
            reason: String explaining why E-stop triggered.

        Returns:
            OrchestratorOutput: Modified container with zeroed velocities.
        """
        self.logger.error(f"EMERGENCY STOP: {reason}", throttle_duration_sec=0.5)
        out.v = 0.0
        out.omega = 0.0
        out.status = f"e_stop: {reason}"
        self.last_cmd_v = 0.0
        self.last_cmd_omega = 0.0
        return out

    def _check_emergency_stops(
        self,
        robot_pose: Pose2D,
        odom: Odometry,
        scan: LaserScan,
        scan_age_sec: float | None,
        visualizer_alive: bool,
        current_time_sec: float,
    ) -> str | None:
        """Evaluates hardware and localization constraints before planning.

        Args:
            robot_pose: Current estimated pose.
            odom: Current odometry message.
            scan: Current Lidar scan.
            scan_age_sec: Seconds since last scan arrival.
            visualizer_alive: Status of external heartbeat.
            imu: Current IMU message.
            current_time_sec: Current system time in seconds.

        Returns:
            str | None: Error message if E-Stop triggered, else None.
        """
        valid_ranges = [
            r
            for r in scan.ranges
            if max(scan.range_min, MIN_SCAN_RANGE) < r < scan.range_max and math.isfinite(r)
        ]
        if valid_ranges and min(valid_ranges) < CRITICAL_STOP_RADIUS:
            return f"Critical Zone Intrusion ({min(valid_ranges):.2f}m)"

        if scan_age_sec is None or scan_age_sec > 1.0:
            return f"LiDAR data stale. Scan age: {scan_age_sec:.2f}s"

        if not visualizer_alive:
            return "Visualizer timeout"

        o = odom.pose.pose.orientation
        roll, pitch, _ = quat_to_euler(o.x, o.y, o.z, o.w)
        if abs(pitch) > MAX_PITCH or abs(roll) > MAX_ROLL:
            return f"Pitch ({pitch:.2f}rad) or Roll ({roll:.2f}rad) limit exceeded"

        if self.last_processed_odom is not None:
            dx, dy, dtheta = get_odom_delta(odom, self.last_processed_odom)
            dist = math.hypot(dx, dy)

            t_curr = odom.header.stamp.sec + (odom.header.stamp.nanosec * 1e-9)
            t_last = self.last_processed_odom.header.stamp.sec + (
                self.last_processed_odom.header.stamp.nanosec * 1e-9
            )
            dt_sec = t_curr - t_last

            if dt_sec > 0.01:
                implied_v = dist / dt_sec
                implied_omega = abs(dtheta) / dt_sec
                if (
                    implied_v > MAX_ODOM_JUMP_SPEED_LINEAR
                    or implied_omega > MAX_ODOM_JUMP_SPEED_ANGULAR
                ):
                    return f"Localization jump: Implied speed v={implied_v:.2f}m/s, w={implied_omega:.2f}rad/s"
            elif dt_sec < 0.0:
                return "Time went backwards, invalidating odometry"

        if self.path_manager.has_goal():
            snapshot = self.path_manager.get_local_path_snapshot()
            idx = self.path_manager.get_current_index()
            if snapshot and idx < len(snapshot):
                px, py = snapshot[idx]
                dist = math.hypot(robot_pose.x - px, robot_pose.y - py)
                if dist > MAX_CROSS_TRACK_ERROR:
                    return f"Extreme Cross-Track Error: {dist:.2f}m off path"

        actual_v = odom.twist.twist.linear.x
        if abs(self.last_cmd_v) > 0.1:
            if abs(actual_v) < STUCK_VEL_TOLERANCE:
                if self.mismatch_start_time is None:
                    self.mismatch_start_time = current_time_sec
                elif (current_time_sec - self.mismatch_start_time) > STUCK_TIMEOUT:
                    return f"Stuck detected. Cmd={self.last_cmd_v:.2f} actual={actual_v:.2f}"
            else:
                self.mismatch_start_time = None
        else:
            self.mismatch_start_time = None

        return None

    def step(
        self,
        odom_g: Odometry | None,
        scan: LaserScan | None,
        scan_age_sec: float | None,
        gps_data: GpsCoord | None,
        first_gps: GpsCoord | None,
        current_time_sec: float,
        visualizer_alive: bool,
        get_latest_odom: Callable[[], Odometry | None],
        performance: PerformanceTracker,
    ) -> OrchestratorOutput:
        """Executes a single algorithmic step of the control loop.

        Args:
            odom_g: Currently cached global odometry.
            scan: Currently cached Lidar scan.
            scan_age_sec: Elapsed seconds since the scan was received.
            gps_data: Currently cached GPS Coordinate.
            first_gps: The initial valid GPS coordinate captured.
            current_time_sec: Current system time in seconds.
            visualizer_alive: Flag verifying external UI/Heartbeat is active.
            get_latest_odom: Lambda to fetch a fresh odometry reading at the end of the loop.
            performance: The active performance tracking context.

        Returns:
            OrchestratorOutput: Result containing all data intended for publication.
        """
        out = OrchestratorOutput()

        if scan is None or odom_g is None or (self.use_gps and gps_data is None):
            self._log_missing_sensors(scan, odom_g, gps_data)
            return out

        dx, dy, dt = get_odom_delta(odom_g, self.last_processed_odom)
        if self.last_processed_odom is not None:
            self.planner.transform_trajectory(dx, dy, dt)
        self.last_processed_odom = odom_g

        o = odom_g.pose.pose.orientation
        robot_pose = Pose2D(
            odom_g.pose.pose.position.x,
            odom_g.pose.pose.position.y,
            quat_to_yaw(o.x, o.y, o.z, o.w),
        )
        out.robot_pose = robot_pose

        self._log_robot_location(robot_pose, gps_data)

        estop_reason = self._check_emergency_stops(
            robot_pose,
            odom_g,
            scan,
            scan_age_sec,
            visualizer_alive,
            current_time_sec,
        )
        if estop_reason:
            return self._trigger_estop(out, estop_reason)

        self.path_manager.update_local_path_from_gps(
            gps_data, robot_pose.x, robot_pose.y
        )
        local_path_snapshot = self.path_manager.get_local_path_snapshot()
        if local_path_snapshot:
            out.global_path = local_path_snapshot

        detected_obstacles = self.obstacle_pipeline.process(scan, SIM)
        if detected_obstacles is None:
            self.logger.warn("Obstacle detection failed", throttle_duration_sec=2.0)
            out.v, out.omega = 0.0, 0.0
            return out

        self.planner.update_obstacles(detected_obstacles)
        out.obstacles = detected_obstacles
        performance.update("Obstacle processing")

        if not self.path_manager.has_goal():
            self.logger.debug("Waiting for a goal...", throttle_duration_sec=2.0)
            out.v, out.omega = 0.0, 0.0
            return out

        if self.use_gps and not first_gps:
            self.logger.warn("Waiting for GPS anchor...", throttle_duration_sec=2.0)
            return out

        self.path_manager.generate_path(gps_data, robot_pose.x, robot_pose.y)

        if self.path_manager.check_goal_reached(robot_pose.x, robot_pose.y):
            self.logger.info("Goal reached!", throttle_duration_sec=2.0)
            out.v, out.omega = 0.0, 0.0
            out.status = "goal_reached"
            return out

        selection = self.goal_selector.select_local_goal(
            path=self.path_manager.get_local_path_snapshot(),
            start_index=self.path_manager.get_current_index(),
            vehicle_x=robot_pose.x,
            vehicle_y=robot_pose.y,
            yaw=robot_pose.theta,
            detected_obstacles=detected_obstacles,
        )
        performance.update("Goal update")

        if selection is None:
            self.logger.error(
                "No local goal could be selected", throttle_duration_sec=2.0
            )
            out.v, out.omega = 0.0, 0.0
            return out

        local_goal, new_closest_idx, target_idx = selection
        if self.use_gps:
            corridor_line_obstacles = self.goal_selector.generate_corridor_barriers(
                path=self.path_manager.get_local_path_snapshot(),
                closest_idx=new_closest_idx,
                target_idx=target_idx,
                vehicle_x=robot_pose.x,
                vehicle_y=robot_pose.y,
                yaw=robot_pose.theta,
                barrier_offset=BARRIER_OFFSET,
            )
            detected_obstacles.extend(corridor_line_obstacles[0])
            detected_obstacles.extend(corridor_line_obstacles[1])

        out.obstacles = detected_obstacles
        self.path_manager.update_current_index(new_closest_idx)
        performance.update("Corridor generation")

        self.planner.move_goal(local_goal, (0.0,))
        astar_waypoints: list[tuple[float, float]] | None = None

        if self.needs_astar_plan:
            astar_waypoints = astar_visibility_path(
                start_xy=(self.planner.start_pose[0], self.planner.start_pose[1]),
                goal_xy=(local_goal[0], local_goal[1]),
                obstacles=detected_obstacles,
                safety_radius=SAFETY_RADIUS,
            )
            if astar_waypoints is None:
                return self._trigger_estop(out, "A* Search Failed")
            self.needs_astar_plan = False

        if self.needs_initial_plan or astar_waypoints is not None:
            self.planner.plan(waypoints=astar_waypoints)
            self.needs_initial_plan = False

        self.planner.refine(
            iterations=MAX_TEB_ITERATIONS,
            current_velocity=odom_g.twist.twist.linear.x,
            current_omega=odom_g.twist.twist.angular.z,
        )

        performance.update("Planner refinement")
        trajectory = self.planner.get_trajectory()

        if trajectory is not None and len(trajectory) > 1:
            for i in range(len(trajectory) - 1):
                if is_segment_safe(
                    float(trajectory[i][0]),
                    float(trajectory[i][1]),
                    float(trajectory[i + 1][0]),
                    float(trajectory[i + 1][1]),
                    detected_obstacles,
                ):
                    continue
                self.logger.warn(
                    f"Trajectory collides at segment {i}; queued A* replan",
                    throttle_duration_sec=1.0,
                )
                self.needs_astar_plan = True
                if i <= TRAJECTORY_COLLISION_LOOKAHEAD:
                    return self._trigger_estop(out, "Imminent Trajectory Collision")

        latest_odom_after = get_latest_odom()

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
                out.robot_pose = robot_pose

        out.trajectory = trajectory
        v, omega = trajectory_to_action(trajectory)

        if abs(v) > MAX_ESTOP_V or abs(omega) > MAX_ESTOP_OMEGA:
            return self._trigger_estop(
                out, f"Limits Exceeded (v={v:.2f}, w={omega:.2f})"
            )

        out.v = v
        out.omega = omega
        out.goal_local = [local_goal[0], local_goal[1], 0.0]
        self.last_cmd_v = v
        self.last_cmd_omega = omega

        return out
