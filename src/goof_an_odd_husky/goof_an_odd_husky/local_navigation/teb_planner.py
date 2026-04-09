from dataclasses import dataclass, field
from typing import override

import numpy as np
from numpy.typing import NDArray
import pyceres

from goof_an_odd_husky.local_navigation.teb_costs import (
    SegmentVelocityCost,
    SegmentAngularVelocityCost,
    SegmentAccelerationCost,
    SegmentAngularAccelerationCost,
    StartAccelerationCost,
    StartAngularAccelerationCost,
    SegmentKinematicsCost,
    SegmentHeadingCost,
    SegmentAngularSmoothingCost,
    SegmentTimeCost,
    SegmentCircleObstaclesCost,
    SegmentLineObstaclesCost,
)
from goof_an_odd_husky_common.helpers import normalize_angle
from goof_an_odd_husky_common.types import Trajectory
from goof_an_odd_husky.local_navigation.trajectory_planner import TrajectoryPlanner
from goof_an_odd_husky_common.obstacles import CircleObstacle, LineObstacle
from goof_an_odd_husky.local_navigation.obstacle_pipeline import ObstacleFilter

DT_MIN: float = 0.01


@dataclass
class TrajectoryState:
    """Encapsulates the three parallel lists that define a TEB trajectory.

    Attributes:
        xy: Per-node 2-D position arrays.
        theta: Per-node heading arrays.
        dt: Per-segment time-step arrays (length = len(xy) - 1).
    """

    xy: list[NDArray[np.float64]] = field(default_factory=list)
    theta: list[NDArray[np.float64]] = field(default_factory=list)
    dt: list[NDArray[np.float64]] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.xy)

    def __bool__(self) -> bool:
        return bool(self.xy)

    def clear(self) -> None:
        """Clear all state lists."""
        self.xy.clear()
        self.theta.clear()
        self.dt.clear()


class TEBPlanner(TrajectoryPlanner):
    """Timed Elastic Band local trajectory planner.

    Optimizes a path utilizing pyceres to enforce kinematics and avoid obstacles.

    Attributes:
        max_v: Maximum linear velocity allowed.
        max_a: Maximum linear acceleration allowed.
        initial_step: Initial distance between trajectory nodes.
        safety_radius: Buffer added around obstacles during optimization.
        traj: Internal trajectory state bound to the Ceres optimizer.
    """

    max_v: float
    max_a: float
    initial_step: float
    safety_radius: float
    traj: TrajectoryState

    def __init__(
        self,
        start_pose: NDArray[np.floating] | list[float],
        goal_pose: NDArray[np.floating] | list[float],
        max_v: float = 0.25,
        max_a: float = 0.1,
        initial_step: float = 1.0,
        safety_radius: float = 1.0,
    ) -> None:
        """Initializes the TEBPlanner.

        Args:
            start_pose: Starting pose [x, y, theta].
            goal_pose: Goal pose [x, y, theta].
            max_v: Maximum linear velocity.
            max_a: Maximum linear acceleration.
            initial_step: Initial spacing distance for trajectory points.
            safety_radius: Minimum allowable distance to obstacles.
        """
        self.setup_poses(start_pose, goal_pose)
        self.max_v = max_v
        self.max_a = max_a
        self.initial_step = initial_step
        self.safety_radius = safety_radius
        self.traj = TrajectoryState()

    @override
    def plan(self) -> Trajectory | None:
        """Generates an initial straight-line trajectory to the goal.

        Returns:
            Trajectory | None: The initial trajectory array Nx4 [x, y, th, dt].
        """
        if np.array_equal(self.start_pose, self.goal_pose):
            return np.array([np.append(self.start_pose, 0.0)])

        dist = np.linalg.norm(self.goal_pose[:2] - self.start_pose[:2])
        n_points = max(2, int(np.ceil(dist / self.initial_step)) + 1)

        xs = np.linspace(self.start_pose[0], self.goal_pose[0], n_points)
        ys = np.linspace(self.start_pose[1], self.goal_pose[1], n_points)

        delta = self.goal_pose[:2] - self.start_pose[:2]
        goal_heading = np.arctan2(delta[1], delta[0])
        dt_init = self.initial_step / self.max_v

        self.traj.clear()

        for i, (x, y) in enumerate(zip(xs, ys)):
            self.traj.xy.append(np.array([x, y], dtype=np.float64))

            if i == 0:
                th = self.start_pose[2]
            elif i == n_points - 1:
                th = self.goal_pose[2]
            else:
                th = goal_heading
            self.traj.theta.append(np.array([th], dtype=np.float64))

            if i < n_points - 1:
                self.traj.dt.append(np.array([dt_init], dtype=np.float64))

        return self.get_trajectory()

    @override
    def get_trajectory(self) -> Trajectory:
        """Construct the full Nx4 trajectory from the current internal state.

        Returns:
            Trajectory: The current trajectory array [x, y, theta, dt].
        """
        if not self.traj:
            return np.array([])

        xy = np.array(self.traj.xy)
        theta = np.array(self.traj.theta)

        if self.traj.dt:
            dts = np.append(np.array(self.traj.dt).flatten(), 0.0).reshape(-1, 1)
        else:
            dts = np.zeros((len(xy), 1))

        return np.hstack((xy, theta, dts))

    @override
    def get_length(self) -> int:
        """Get the number of points in the current trajectory.

        Returns:
            int: Trajectory length.
        """
        return len(self.traj)

    def resize_trajectory(self, min_distance: float, max_distance: float) -> None:
        """Dynamically adjusts trajectory resolution by inserting or removing nodes.

        Ensures the trajectory always has at least 3 points (Start, Mid, Goal)
        to prevent segfaults in Ceres caused by all-constant parameter blocks.

        Args:
            min_distance: Nodes closer than this are pruned.
            max_distance: Nodes further apart than this get a midpoint inserted.
        """
        if not self.traj:
            return

        if len(self.traj) == 2:
            mid_xy = (self.traj.xy[0] + self.traj.xy[1]) / 2.0

            th_start = float(self.traj.theta[0][0])
            th_goal = float(self.traj.theta[1][0])
            mid_th = normalize_angle(
                th_start + normalize_angle(th_goal - th_start) / 2.0
            )

            if self.traj.dt:
                half_dt = float(self.traj.dt[0][0]) / 2.0
                self.traj.dt[0] = np.array([half_dt], dtype=np.float64)
                self.traj.dt.insert(1, np.array([half_dt], dtype=np.float64))
            else:
                self.traj.dt = [np.array([1.0]), np.array([1.0])]

            self.traj.xy.insert(1, np.array(mid_xy, dtype=np.float64))
            self.traj.theta.insert(1, np.array([mid_th], dtype=np.float64))

        i = 0
        while i < len(self.traj) - 1:
            dist = np.linalg.norm(self.traj.xy[i + 1] - self.traj.xy[i])

            if dist > max_distance:
                mid_xy = (self.traj.xy[i] + self.traj.xy[i + 1]) / 2.0
                th_curr = float(self.traj.theta[i][0])
                th_next = float(self.traj.theta[i + 1][0])
                mid_th = normalize_angle(
                    th_curr + normalize_angle(th_next - th_curr) / 2.0
                )

                half_dt = float(self.traj.dt[i][0]) / 2.0
                self.traj.dt[i] = np.array([half_dt], dtype=np.float64)
                self.traj.xy.insert(i + 1, np.array(mid_xy, dtype=np.float64))
                self.traj.theta.insert(i + 1, np.array([mid_th], dtype=np.float64))
                self.traj.dt.insert(i + 1, np.array([half_dt], dtype=np.float64))
                continue

            elif dist < min_distance:
                if len(self.traj) <= 3:
                    i += 1
                    continue

                if i == 0:
                    remove_idx = i + 1
                    new_dt = float(self.traj.dt[i][0] + self.traj.dt[i + 1][0])
                    self.traj.dt[i] = np.array([new_dt], dtype=np.float64)
                    self.traj.dt.pop(i + 1)
                else:
                    remove_idx = i
                    new_dt = float(self.traj.dt[i - 1][0] + self.traj.dt[i][0])
                    self.traj.dt[i - 1] = np.array([new_dt], dtype=np.float64)
                    self.traj.dt.pop(i)
                    i -= 1

                self.traj.xy.pop(remove_idx)
                self.traj.theta.pop(remove_idx)
                continue

            i += 1

    @override
    def move_goal(self, new_xy: tuple[float, float], new_theta: tuple[float]) -> None:
        """Move the goal point.

        Args:
            new_xy: (x, y) coordinate tuple of the new goal.
            new_theta: A tuple containing the orientation.
        """
        self.goal_pose = np.array([*new_xy, *new_theta])

        if not self.traj:
            return

        self.traj.xy[-1][:] = np.asarray(new_xy).reshape(2)
        self.traj.theta[-1][:] = np.asarray(new_theta).reshape(1)

    @override
    def get_distance_goal(self) -> float:
        """Return the distance to the goal.

        Returns:
            float: The straight line distance from start to the goal.
        """
        return np.linalg.norm(self.traj.xy[-1] - self.traj.xy[0]).item()

    @override
    def transform_trajectory(self, dx: float, dy: float, dtheta: float) -> None:
        """Transforms the internal trajectory guess to account for robot motion.

        Args:
            dx: Change in robot position X relative to the previous frame.
            dy: Change in robot position Y.
            dtheta: Change in robot heading.
        """
        if not self.traj:
            return

        c = np.cos(-dtheta)
        s = np.sin(-dtheta)
        rot_mat = np.array([[c, -s], [s, c]])
        translation = np.array([dx, dy])

        for i in range(len(self.traj)):
            self.traj.xy[i][:] = rot_mat @ (self.traj.xy[i] - translation)

        for i in range(len(self.traj)):
            self.traj.theta[i][:] = normalize_angle(self.traj.theta[i][0] - dtheta)

        self.traj.xy[0][:] = np.array([0.0, 0.0])
        self.traj.theta[0][:] = np.array([0.0])

    @override
    def refine(
        self,
        iterations: int = 10,
        current_velocity: float = 0.0,
        current_omega: float = 0.0,
    ) -> bool:
        """Execute the Ceres solver optimization step to smooth the trajectory.

        Args:
            iterations: Maximum solver iterations.
            current_velocity: The starting linear velocity in m/s.
            current_omega: The starting angular velocity in rad/s.

        Returns:
            bool: True if the solver converged successfully, False otherwise.
        """
        if not self.traj:
            return False

        self.resize_trajectory(0.4, 6)

        if len(self.traj) <= 2:
            return False

        problem = pyceres.Problem()

        circle_obstacles = [o for o in self.obstacles if isinstance(o, CircleObstacle)]
        line_obstacles = [o for o in self.obstacles if isinstance(o, LineObstacle)]

        start_theta = self.traj.theta[0]
        start_vx = current_velocity * np.cos(start_theta).item()
        start_vy = current_velocity * np.sin(start_theta).item()

        velocity_cost = SegmentVelocityCost(weight=250.0, max_v=self.max_v)
        angular_velocity_cost = SegmentAngularVelocityCost(
            weight=90.0, max_omega=self.max_v / 2
        )
        acceleration_cost = SegmentAccelerationCost(weight=90.0, max_a=self.max_a)
        angular_acceleration_cost = SegmentAngularAccelerationCost(
            weight=50.0, max_alpha=self.max_a / 3
        )
        start_acceleration_cost = StartAccelerationCost(
            weight=90.0, max_a=self.max_a, current_v=(start_vx, start_vy)
        )
        start_angular_acceleration_cost = StartAngularAccelerationCost(
            weight=50.0, max_alpha=self.max_a / 3, current_omega=current_omega
        )
        kinematic_cost = SegmentKinematicsCost(weight=9.0)
        heading_cost = SegmentHeadingCost(weight=40.0)
        angular_smoothing_cost = SegmentAngularSmoothingCost(weight=4.0)
        time_cost = SegmentTimeCost(weight=1.0)

        obstacle_filter = ObstacleFilter(
            circle_obstacles, line_obstacles, self.safety_radius * 3.0
        )

        n_points = len(self.traj)

        for i in range(n_points - 1):
            xy_curr = self.traj.xy[i]
            xy_next = self.traj.xy[i + 1]
            dt = self.traj.dt[i]
            theta_curr = self.traj.theta[i]
            theta_next = self.traj.theta[i + 1]

            close_circles = obstacle_filter.get_close_circles(
                xy_curr[0], xy_curr[1], xy_next[0], xy_next[1]
            )
            if close_circles:
                problem.add_residual_block(
                    SegmentCircleObstaclesCost(
                        close_circles, weight=200.0, safety_radius=self.safety_radius
                    ),
                    None,
                    [xy_curr, xy_next],
                )

            close_lines = obstacle_filter.get_close_lines(
                xy_curr[0], xy_curr[1], xy_next[0], xy_next[1]
            )
            if close_lines:
                problem.add_residual_block(
                    SegmentLineObstaclesCost(
                        close_lines, weight=200.0, safety_radius=self.safety_radius
                    ),
                    None,
                    [xy_curr, xy_next],
                )

            problem.add_residual_block(velocity_cost, None, [xy_curr, xy_next, dt])
            problem.add_residual_block(
                angular_velocity_cost, None, [theta_curr, theta_next, dt]
            )

            if i < n_points - 2:
                problem.add_residual_block(
                    acceleration_cost,
                    None,
                    [xy_curr, xy_next, self.traj.xy[i + 2], dt, self.traj.dt[i + 1]],
                )
                problem.add_residual_block(
                    angular_acceleration_cost,
                    None,
                    [
                        theta_curr,
                        theta_next,
                        self.traj.theta[i + 2],
                        dt,
                        self.traj.dt[i + 1],
                    ],
                )

            if i == 0:
                problem.add_residual_block(
                    start_acceleration_cost, None, [xy_curr, xy_next, dt]
                )
                problem.add_residual_block(
                    start_angular_acceleration_cost, None, [theta_curr, theta_next, dt]
                )

            problem.add_residual_block(
                kinematic_cost, None, [xy_curr, theta_curr, xy_next, theta_next]
            )
            problem.add_residual_block(time_cost, None, [dt])
            problem.add_residual_block(
                heading_cost, None, [xy_curr, theta_curr, xy_next]
            )
            problem.add_residual_block(
                angular_smoothing_cost, None, [theta_curr, theta_next, dt]
            )

            problem.set_parameter_lower_bound(dt, 0, DT_MIN)

        problem.set_parameter_block_constant(self.traj.xy[0])
        problem.set_parameter_block_constant(self.traj.xy[-1])
        problem.set_parameter_block_constant(self.traj.theta[0])
        problem.set_parameter_block_constant(self.traj.theta[-1])

        options = pyceres.SolverOptions()
        options.max_num_iterations = iterations
        options.function_tolerance = 1e-4
        options.gradient_tolerance = 1e-4
        options.parameter_tolerance = 1e-4
        options.num_threads = 4
        options.linear_solver_type = pyceres.LinearSolverType.SPARSE_NORMAL_CHOLESKY
        options.minimizer_progress_to_stdout = False

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        return summary.termination_type == pyceres.TerminationType.CONVERGENCE
