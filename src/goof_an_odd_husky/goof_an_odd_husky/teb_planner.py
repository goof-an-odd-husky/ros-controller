from goof_an_odd_husky.teb_costs import (
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
from goof_an_odd_husky.helpers import normalize_angle
from typing import override
from goof_an_odd_husky.trajectory_planner import TrajectoryPlanner
from goof_an_odd_husky.obstacles import (
    CircleObstacle,
    LineObstacle,
    ObstacleFilter,
)
import numpy as np
from numpy.typing import NDArray
import pyceres

DT_MIN = 0.01


class TEBPlanner(TrajectoryPlanner):
    def __init__(
        self,
        start_pose: NDArray[np.floating] | list[float],
        goal_pose: NDArray[np.floating] | list[float],
        max_v: float,
        max_a: float,
        initial_step: float = 3.0,
        safety_radius: float = 2.0,
    ):
        self.setup_poses(start_pose, goal_pose)
        self.max_v = max_v
        self.max_a = max_a
        self.initial_step = initial_step
        self.safety_radius = safety_radius

        # pyceres binds to the memory of these specific numpy arrays
        # therefore we have save lists long-term
        self.optimization_xy = []
        self.optimization_theta = []
        self.optimization_dt = []

    @override
    def plan(self) -> NDArray[np.floating] | None:
        if np.array_equal(self.start_pose, self.goal_pose):
            return np.array([np.append(self.start_pose, 0.0)])

        dist = np.linalg.norm(self.goal_pose[:2] - self.start_pose[:2])
        n_points = max(2, int(np.ceil(dist / self.initial_step)) + 1)

        xs = np.linspace(self.start_pose[0], self.goal_pose[0], n_points)
        ys = np.linspace(self.start_pose[1], self.goal_pose[1], n_points)

        delta = self.goal_pose[:2] - self.start_pose[:2]
        goal_heading = np.arctan2(delta[1], delta[0])

        dt_init = self.initial_step / self.max_v

        self.optimization_xy = []
        self.optimization_theta = []
        self.optimization_dt = []

        for i, (x, y) in enumerate(zip(xs, ys)):
            self.optimization_xy.append(np.array([x, y], dtype=np.float64))

            if i == 0:
                th = self.start_pose[2]
            elif i == n_points - 1:
                th = self.goal_pose[2]
            else:
                th = goal_heading
            self.optimization_theta.append(np.array([th], dtype=np.float64))

            if i < n_points - 1:
                self.optimization_dt.append(np.array([dt_init], dtype=np.float64))

        return self.get_trajectory()

    def get_trajectory(self):
        if not self.optimization_xy:
            return np.array([])

        xy = np.array(self.optimization_xy)
        theta = np.array(self.optimization_theta)

        if self.optimization_dt:
            dts = np.array(self.optimization_dt).flatten()
            dts = np.append(dts, 0.0).reshape(-1, 1)
        else:
            dts = np.zeros((len(xy), 1))

        return np.hstack((xy, theta, dts))

    def get_length(self):
        return len(self.optimization_xy)

    def resize_trajectory(self, min_distance: float, max_distance: float):
        """
        Dynamically adjusts the resolution of the trajectory.
        Ensures the trajectory always has at least 3 points (Start, Mid, Goal)
        to prevent segfaults in Ceres Solver caused by all-constant parameter blocks.
        """
        if not self.optimization_xy:
            return

        if len(self.optimization_xy) == 2:
            xy_start = self.optimization_xy[0]
            xy_goal = self.optimization_xy[1]

            mid_xy = (xy_start + xy_goal) / 2.0

            th_start = self.optimization_theta[0]
            th_goal = self.optimization_theta[1]
            diff = normalize_angle(th_goal - th_start)
            mid_th = normalize_angle(th_start + diff / 2.0)

            if self.optimization_dt:
                dt_total = self.optimization_dt[0]
                half_dt = dt_total / 2.0
                self.optimization_dt[0] = np.array([half_dt.item()], dtype=np.float64)
                self.optimization_dt.insert(
                    1, np.array([half_dt.item()], dtype=np.float64)
                )
            else:
                self.optimization_dt = [np.array([1.0]), np.array([1.0])]

            self.optimization_xy.insert(1, np.array(mid_xy, dtype=np.float64))
            self.optimization_theta.insert(
                1, np.array([mid_th.item()], dtype=np.float64)
            )

        i = 0
        while i < len(self.optimization_xy) - 1:
            xy_curr = self.optimization_xy[i]
            xy_next = self.optimization_xy[i + 1]
            dist = np.linalg.norm(xy_next - xy_curr)

            if dist > max_distance:
                mid_xy = (xy_curr + xy_next) / 2.0

                th_curr = self.optimization_theta[i]
                th_next = self.optimization_theta[i + 1]
                diff = normalize_angle(th_next - th_curr)
                mid_th = normalize_angle(th_curr + diff / 2.0)

                current_dt = self.optimization_dt[i]
                half_dt = current_dt / 2.0

                self.optimization_dt[i] = np.array([half_dt.item()], dtype=np.float64)

                self.optimization_xy.insert(i + 1, np.array(mid_xy, dtype=np.float64))
                self.optimization_theta.insert(
                    i + 1, np.array([mid_th.item()], dtype=np.float64)
                )
                self.optimization_dt.insert(
                    i + 1, np.array([half_dt.item()], dtype=np.float64)
                )

                continue

            elif dist < min_distance:
                if len(self.optimization_xy) <= 3:
                    i += 1
                    continue

                if i == 0:
                    remove_idx = i + 1
                    new_dt = self.optimization_dt[i] + self.optimization_dt[i + 1]
                    self.optimization_dt[i] = np.array(
                        [new_dt.item()], dtype=np.float64
                    )
                    self.optimization_dt.pop(i + 1)
                else:
                    remove_idx = i
                    new_dt = self.optimization_dt[i - 1] + self.optimization_dt[i]
                    self.optimization_dt[i - 1] = np.array(
                        [new_dt.item()], dtype=np.float64
                    )
                    self.optimization_dt.pop(i)
                    i -= 1

                self.optimization_xy.pop(remove_idx)
                self.optimization_theta.pop(remove_idx)

                continue

            i += 1

    def move_goal(
        self,
        new_xy: tuple[float, float],
        new_theta: tuple[float],
    ) -> None:
        """
        Move the goal point.
        """
        self.goal_pose = np.array([*new_xy, *new_theta])

        if not self.optimization_xy:
            return

        self.optimization_xy[-1][:] = np.asarray(new_xy).reshape(2)
        self.optimization_theta[-1][:] = np.asarray(new_theta).reshape(1)

    @override
    def get_distance_goal(self) -> float:
        """Return the distance to the goal.

        Returns:
            The straight line distance to the goal.
        """
        return np.linalg.norm(self.optimization_xy[-1] - self.optimization_xy[0]).item()

    def transform_trajectory(self, dx: float, dy: float, dtheta: float):
        """Transforms the internal trajectory guess to account for robot motion.

        This keeps the trajectory aligned with the world while the robot frame moves.

        Args:
            dx, dy: Change in robot position relative to the previous frame's frame.
            dtheta: Change in robot heading.
        """
        if not self.optimization_xy:
            return

        c = np.cos(-dtheta)
        s = np.sin(-dtheta)
        rot_mat = np.array([[c, -s], [s, c]])

        translation = np.array([dx, dy])

        for i in range(len(self.optimization_xy)):
            p_old = self.optimization_xy[i]

            p_shifted = p_old - translation

            p_new = rot_mat @ p_shifted

            self.optimization_xy[i][:] = p_new

        for i in range(len(self.optimization_theta)):
            th_old = self.optimization_theta[i][0]
            th_new = normalize_angle(th_old - dtheta)
            self.optimization_theta[i][:] = np.array([th_new])

        self.optimization_xy[0][:] = np.array([0.0, 0.0])
        self.optimization_theta[0][:] = np.array([0.0])

    @override
    def refine(
        self,
        iterations: int = 10,
        current_velocity: float = 0,
        current_omega: float = 0,
    ) -> bool:
        if not self.optimization_xy:
            return False

        self.resize_trajectory(1, 6)  # todo: make into constants

        if len(self.optimization_xy) <= 2:
            return False

        problem = pyceres.Problem()

        circle_obstacles = [
            obs for obs in self.obstacles if isinstance(obs, CircleObstacle)
        ]
        line_obstacles = [
            obs for obs in self.obstacles if isinstance(obs, LineObstacle)
        ]

        start_theta = self.optimization_theta[0]
        start_vx = current_velocity * np.cos(start_theta).item()
        start_vy = current_velocity * np.sin(start_theta).item()
        current_velocity_vec = (start_vx, start_vy)

        velocity_cost = SegmentVelocityCost(weight=10.0, max_v=self.max_v)
        angular_velocity_cost = SegmentAngularVelocityCost(
            weight=10.0, max_omega=self.max_v / 2
        )
        acceleration_cost = SegmentAccelerationCost(weight=10.0, max_a=self.max_a)
        angular_acceleration_cost = SegmentAngularAccelerationCost(
            weight=10.0, max_alpha=self.max_a / 3
        )
        start_acceleration_cost = StartAccelerationCost(
            weight=10.0, max_a=self.max_a, current_v=current_velocity_vec
        )
        start_angular_acceleration_cost = StartAngularAccelerationCost(
            weight=10.0, max_alpha=self.max_a / 3, current_omega=current_omega
        )
        kinematic_cost = SegmentKinematicsCost(weight=7.0)
        heading_cost = SegmentHeadingCost(weight=25.0)
        angular_smoothing_cost = SegmentAngularSmoothingCost(weight=1)
        time_cost = SegmentTimeCost(weight=10.0)

        obstacle_filter = ObstacleFilter(
            circle_obstacles, line_obstacles, self.safety_radius * 2.5
        )

        n_points = len(self.optimization_xy)

        for i in range(n_points - 1):
            xy_curr = self.optimization_xy[i]
            xy_next = self.optimization_xy[i + 1]
            dt = self.optimization_dt[i]

            theta_curr = self.optimization_theta[i]
            theta_next = self.optimization_theta[i + 1]

            close_circles = obstacle_filter.get_close_circles(
                xy_curr[0], xy_curr[1], xy_next[0], xy_next[1]
            )
            if close_circles:
                circle_cost = SegmentCircleObstaclesCost(
                    close_circles, weight=50.0, safety_radius=self.safety_radius
                )
                problem.add_residual_block(circle_cost, None, [xy_curr, xy_next])

            close_lines = obstacle_filter.get_close_lines(
                xy_curr[0], xy_curr[1], xy_next[0], xy_next[1]
            )
            if close_lines:
                line_cost = SegmentLineObstaclesCost(
                    close_lines, weight=50.0, safety_radius=self.safety_radius
                )
                problem.add_residual_block(line_cost, None, [xy_curr, xy_next])

            problem.add_residual_block(velocity_cost, None, [xy_curr, xy_next, dt])
            problem.add_residual_block(
                angular_velocity_cost, None, [theta_curr, theta_next, dt]
            )

            if i < n_points - 2:
                problem.add_residual_block(
                    acceleration_cost,
                    None,
                    [
                        xy_curr,
                        xy_next,
                        self.optimization_xy[i + 2],
                        dt,
                        self.optimization_dt[i + 1],
                    ],
                )
                problem.add_residual_block(
                    angular_acceleration_cost,
                    None,
                    [
                        theta_curr,
                        theta_next,
                        self.optimization_theta[i + 2],
                        dt,
                        self.optimization_dt[i + 1],
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

        problem.set_parameter_block_constant(self.optimization_xy[0])
        problem.set_parameter_block_constant(self.optimization_xy[-1])
        problem.set_parameter_block_constant(self.optimization_theta[0])
        problem.set_parameter_block_constant(self.optimization_theta[-1])

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
