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
    MAX_A,
    INITIAL_STEP,
    CLUSTER_BREAK_DISTANCE,
    GEOMETRY_SPLIT_THRESHOLD,
    MIN_SCAN_RANGE,
    MEDIAN_FILTER_SIZE,
    SOFTMIN_ALPHA,
    TRAJECTORY_LIMITS,
    TEB_WEIGHTS,
    OSM_RELATION_ID,
    MIN_CIRCLE_RADIUS,
    MAX_CIRCLE_RADIUS,
    MAX_LINE_DISTANCE,
)
from goof_an_odd_husky_common.helpers import quat_to_yaw
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
        """Helper to log warnings when required sensors are absent."""
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
        """Helper to log the current robot location."""
        if self.use_gps and gps_data:
            msg = (
                f"GPS: Lat={gps_data.lat:.6f}, "
                f"Lon={gps_data.lon:.6f}, "
                f"Yaw={robot_pose.theta:.2f}rad"
            )
        else:
            msg = f"Pose: X={robot_pose.x:.2f}m, Y={robot_pose.y:.2f}m, Yaw={robot_pose.theta:.2f}rad"
        self.logger.info(msg, throttle_duration_sec=2.0)

    def step(
        self,
        odom_g: Odometry | None,
        scan: LaserScan | None,
        scan_age_sec: float | None,
        gps_data: GpsCoord | None,
        first_gps: GpsCoord | None,
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

        if not visualizer_alive:
            self.logger.warn("Waiting for visualizer...", throttle_duration_sec=2.0)
            out.v, out.omega = 0.0, 0.0
            return out

        self.path_manager.update_local_path_from_gps(
            gps_data, robot_pose.x, robot_pose.y
        )
        local_path_snapshot = self.path_manager.get_local_path_snapshot()
        if local_path_snapshot:
            out.global_path = local_path_snapshot

        if scan_age_sec is None or scan_age_sec > 1.0:
            self.logger.error("Lidar data stale", throttle_duration_sec=1.0)
            out.v, out.omega = 0.0, 0.0
            return out

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
            self.needs_astar_plan = False

        if self.needs_initial_plan or astar_waypoints is not None:
            self.planner.plan(waypoints=astar_waypoints)
            self.needs_initial_plan = False

        self.planner.refine(
            current_velocity=odom_g.twist.twist.linear.x,
            current_omega=odom_g.twist.twist.angular.z,
        )

        performance.update("Planner refinement")
        trajectory = self.planner.get_trajectory()

        if (
            trajectory is not None
            and len(trajectory) > 1
            and any(
                not is_segment_safe(
                    float(trajectory[i][0]),
                    float(trajectory[i][1]),
                    float(trajectory[i + 1][0]),
                    float(trajectory[i + 1][1]),
                    detected_obstacles,
                )
                for i in range(len(trajectory) - 1)
            )
        ):
            self.needs_astar_plan = True
            self.logger.warn(
                "Trajectory collides; queued A* replan", throttle_duration_sec=1.0
            )

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
        out.v = v
        out.omega = omega
        out.goal_local = [local_goal[0], local_goal[1], 0.0]

        return out
