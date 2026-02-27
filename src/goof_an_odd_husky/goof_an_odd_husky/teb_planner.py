from goof_an_odd_husky.helpers import normalize_angle
from typing import override
from .trajectory_planner import TrajectoryPlanner
import numpy as np
from numpy.typing import NDArray
import pyceres


DT_MIN = 0.01


class SegmentObstaclesCost(pyceres.CostFunction):
    def __init__(
        self, obstacles: NDArray[np.floating], weight: float, safety_radius: float
    ):
        super().__init__()
        self.obstacles_x = obstacles[
            :, 0
        ]  # todo: optimize and use separate arrays from the start?
        self.obstacles_y = obstacles[:, 1]
        self.obstacles_r = obstacles[:, 2]
        self.n_obstacles = len(self.obstacles_r)
        self.weight = weight
        self.safety_radius = safety_radius

        self.set_num_residuals(self.n_obstacles)
        self.set_parameter_block_sizes([2, 2])

    def Evaluate(self, parameters, residuals, jacobians):
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len_sq = max(AB_x**2 + AB_y**2, 1e-10)

        # O - obstacle
        AO_x = self.obstacles_x - A_x
        AO_y = self.obstacles_y - A_y

        # AO1 - projection of AO onto AB
        # t = |AO1|/|AB|
        t = (AO_x * AB_x + AO_y * AB_y) / AB_len_sq
        t = np.clip(t, 0.0, 1.0)

        O1O_x = AO_x - t * AB_x
        O1O_y = AO_y - t * AB_y

        O1O_len_sq = O1O_x**2 + O1O_y**2
        O1O_len = np.sqrt(O1O_len_sq + 1e-10)

        errors = self.obstacles_r + self.safety_radius - O1O_len
        mask = errors > 0.0
        w = self.weight
        residuals[:] = np.where(mask, w * errors, 0.0)

        if jacobians is not None:
            # d_hat = O1O / |O1O|
            inv_dist = 1.0 / O1O_len
            d_hat_x = O1O_x * inv_dist
            d_hat_y = O1O_y * inv_dist

            # error < 0 => jacobian = 0
            j_scaler = np.where(mask, w, 0.0)

            if jacobians[0] is not None:
                j_A_factor = j_scaler * (1.0 - t)
                J_A_x = j_A_factor * d_hat_x
                J_A_y = j_A_factor * d_hat_y

                jacobians[0][:] = np.vstack((J_A_x, J_A_y)).T.ravel()

            if jacobians[1] is not None:
                j_B_factor = j_scaler * t
                J_B_x = j_B_factor * d_hat_x
                J_B_y = j_B_factor * d_hat_y

                jacobians[1][:] = np.vstack((J_B_x, J_B_y)).T.ravel()

        return True


class SegmentVelocityCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_v: float):
        super().__init__()
        self.weight = weight
        self.max_v = max_v

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]
        dt = parameters[2][0]
        if abs(dt) < 1e-9:
            return False
        inv_dt = 1 / dt

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len = np.sqrt(AB_x**2 + AB_y**2 + 1e-10)

        v = AB_len * inv_dt

        diff = 0
        if v > self.max_v:
            diff = v - self.max_v

        w = self.weight
        residuals[0] = w * diff

        if jacobians is not None:
            if v <= self.max_v:
                if jacobians[0] is not None:
                    jacobians[0][:] = [0.0, 0.0]
                if jacobians[1] is not None:
                    jacobians[1][:] = [0.0, 0.0]
                if jacobians[2] is not None:
                    jacobians[2][0] = 0.0
                return True

            dr_dv = w
            scale = dr_dv / (AB_len * dt)

            if jacobians[0] is not None:
                jacobians[0][0] = -AB_x * scale
                jacobians[0][1] = -AB_y * scale

            if jacobians[1] is not None:
                jacobians[1][0] = AB_x * scale
                jacobians[1][1] = AB_y * scale

            if jacobians[2] is not None:
                # dv/dt = -AB_len / dt^2 = -v / dt
                dv_dt = -v * inv_dt
                jacobians[2][0] = dr_dv * dv_dt

        return True


class SegmentAngularVelocityCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_omega: float):
        super().__init__()
        self.weight = weight
        self.max_omega = max_omega

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        A_theta = parameters[0][0]
        B_theta = parameters[1][0]
        dt = parameters[2][0]
        if abs(dt) < 1e-9:
            return False
        inv_dt = 1 / dt

        delta_theta = normalize_angle(B_theta - A_theta)

        omega = delta_theta * inv_dt

        diff = 0
        if omega > self.max_omega:
            diff = omega - self.max_omega
        elif omega < -self.max_omega:
            diff = omega + self.max_omega

        w = self.weight
        residuals[0] = w * diff

        if jacobians is not None:
            if omega <= self.max_omega:
                if jacobians[0] is not None:
                    jacobians[0][0] = 0.0
                if jacobians[1] is not None:
                    jacobians[1][0] = 0.0
                if jacobians[2] is not None:
                    jacobians[2][0] = 0.0
                return True

            dr_ddelta_theta = w * inv_dt

            if jacobians[0] is not None:
                jacobians[0][0] = -dr_ddelta_theta

            if jacobians[1] is not None:
                jacobians[1][0] = dr_ddelta_theta

            if jacobians[2] is not None:
                jacobians[2][0] = -w * omega * inv_dt

        return True


class SegmentAccelerationCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_a: float):
        super().__init__()
        self.weight = weight
        self.max_a = max_a

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 2, 1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]
        C_x, C_y = parameters[2]
        dt1 = parameters[3][0]
        dt2 = parameters[4][0]
        if abs(dt1) < 1e-9 or abs(dt2) < 1e-9:
            return False

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len = np.sqrt(AB_x**2 + AB_y**2 + 1e-10)

        BC_x = C_x - B_x
        BC_y = C_y - B_y
        BC_len = np.sqrt(BC_x**2 + BC_y**2 + 1e-10)

        dt = dt1 + dt2

        a = (BC_len / dt2 - AB_len / dt1) * 2 / dt

        diff = 0
        abs_a = abs(a)
        if abs_a > self.max_a:
            diff = abs_a - self.max_a
            sign_a = 1.0 if a > 0 else -1.0
        else:
            sign_a = 0.0

        w = self.weight
        residuals[0] = w * diff

        if jacobians is not None:
            if abs_a <= self.max_a:
                if jacobians[0] is not None:
                    jacobians[0][:] = [0.0, 0.0]
                if jacobians[1] is not None:
                    jacobians[1][:] = [0.0, 0.0]
                if jacobians[2] is not None:
                    jacobians[2][:] = [0.0, 0.0]
                if jacobians[3] is not None:
                    jacobians[3][0] = 0.0
                if jacobians[4] is not None:
                    jacobians[4][0] = 0.0
                return True

            dt1_sq = dt1 * dt1
            dt2_sq = dt2 * dt2
            dt_sq = dt * dt

            dx1 = B_x - A_x
            dx2 = C_x - B_x
            dy1 = B_y - A_y
            dy2 = C_y - B_y

            w = w * sign_a

            l1_dt1 = AB_len * dt1
            l2_dt2 = BC_len * dt2
            dt_half = dt * 0.5
            dt_half_1 = l1_dt1 * dt_half
            inv1 = w / dt_half_1
            dt_half_2 = l2_dt2 * dt_half
            inv2 = w / dt_half_2
            dt_half_12 = l2_dt2 * dt_half_1
            inv12 = w / dt_half_12

            if jacobians[0] is not None:
                jacobians[0][:] = [dx1 * inv1, dy1 * inv1]
            if jacobians[1] is not None:
                jacobians[1][:] = [
                    -(l2_dt2 * dx1 + l1_dt1 * dx2) * inv12,
                    -(l2_dt2 * dy1 + l1_dt1 * dy2) * inv12,
                ]
            if jacobians[2] is not None:
                jacobians[2][:] = [dx2 * inv2, dy2 * inv2]
            if jacobians[3] is not None:
                jacobians[3][0] = (
                    2
                    * w
                    * (-BC_len * dt1_sq + AB_len * dt2 * (dt + dt1))
                    / (dt1_sq * dt2 * dt_sq)
                )
            if jacobians[4] is not None:
                jacobians[4][0] = (
                    2
                    * w
                    * (AB_len * dt2_sq - BC_len * dt1 * (dt + dt2))
                    / (dt2_sq * dt1 * dt_sq)
                )

        return True


class SegmentAngularAccelerationCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_alpha: float):
        super().__init__()
        self.weight = weight
        self.max_alpha = max_alpha

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1, 1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        A_theta = parameters[0][0]
        B_theta = parameters[1][0]
        C_theta = parameters[2][0]
        dt1 = parameters[3][0]
        dt2 = parameters[4][0]

        if abs(dt1) < 1e-9 or abs(dt2) < 1e-9:
            return False

        dt = dt1 + dt2
        inv_dt = 1.0 / dt
        inv_dt1 = 1.0 / dt1
        inv_dt2 = 1.0 / dt2

        delta_theta1 = normalize_angle(B_theta - A_theta)
        delta_theta2 = normalize_angle(C_theta - B_theta)

        omega1 = delta_theta1 * inv_dt1
        omega2 = delta_theta2 * inv_dt2

        alpha = 2.0 * (omega2 - omega1) * inv_dt

        diff = 0.0
        if alpha > self.max_alpha:
            diff = alpha - self.max_alpha
        elif alpha < -self.max_alpha:
            diff = alpha + self.max_alpha

        w = self.weight
        residuals[0] = w * diff

        if jacobians is not None:
            if diff == 0.0:
                if jacobians[0] is not None:
                    jacobians[0][0] = 0.0
                if jacobians[1] is not None:
                    jacobians[1][0] = 0.0
                if jacobians[2] is not None:
                    jacobians[2][0] = 0.0
                if jacobians[3] is not None:
                    jacobians[3][0] = 0.0
                if jacobians[4] is not None:
                    jacobians[4][0] = 0.0
                return True

            factor = 2.0 * w * inv_dt

            if jacobians[0] is not None:
                jacobians[0][0] = factor * inv_dt1

            if jacobians[1] is not None:
                jacobians[1][0] = -factor * (inv_dt1 + inv_dt2)

            if jacobians[2] is not None:
                jacobians[2][0] = factor * inv_dt2

            if jacobians[3] is not None:
                jacobians[3][0] = factor * omega1 * inv_dt1 - w * alpha * inv_dt

            if jacobians[4] is not None:
                jacobians[4][0] = -factor * omega2 * inv_dt2 - w * alpha * inv_dt

        return True


class SegmentKinematicsCost(pyceres.CostFunction):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        x1, y1 = parameters[0]
        angle1 = parameters[1][0]
        x2, y2 = parameters[2]
        angle2 = parameters[3][0]

        c1, s1 = np.cos(angle1), np.sin(angle1)
        c2, s2 = np.cos(angle2), np.sin(angle2)

        dx = x2 - x1
        dy = y2 - y1

        cos_sum = c1 + c2
        sin_sum = s1 + s2

        error = cos_sum * dy - sin_sum * dx

        w = self.weight
        residuals[:] = w * error

        if jacobians is not None:
            w_sin_sum = w * sin_sum
            w_cos_sum = w * cos_sum

            if jacobians[0] is not None:
                jacobians[0][:] = [w_sin_sum, -w_cos_sum]

            if jacobians[1] is not None:
                term = -(dy * s1 + dx * c1)
                jacobians[1][:] = w * term

            if jacobians[2] is not None:
                jacobians[2][:] = [-w_sin_sum, w_cos_sum]

            if jacobians[3] is not None:
                term = -(dy * s2 + dx * c2)
                jacobians[3][:] = w * term

        return True


class SegmentHeadingCost(pyceres.CostFunction):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2])

    def Evaluate(self, parameters, residuals, jacobians):
        x1, y1 = parameters[0]
        theta1 = parameters[1][0]
        x2, y2 = parameters[2]

        dx = x2 - x1
        dy = y2 - y1

        c1, s1 = np.cos(theta1), np.sin(theta1)

        dot = dx * c1 + dy * s1

        if dot >= 0:
            residuals[0] = 0.0
            if jacobians is not None:
                if jacobians[0] is not None:
                    jacobians[0][:] = 0.0
                if jacobians[1] is not None:
                    jacobians[1][:] = 0.0
                if jacobians[2] is not None:
                    jacobians[2][:] = 0.0
            return True

        w = self.weight
        residuals[0] = w * (-dot)

        if jacobians is not None:
            if jacobians[0] is not None:
                jacobians[0][0] = w * c1
                jacobians[0][1] = w * s1

            if jacobians[1] is not None:
                jacobians[1][0] = w * (dx * s1 - dy * c1)

            if jacobians[2] is not None:
                jacobians[2][0] = -w * c1
                jacobians[2][1] = -w * s1

        return True


class SegmentAngularSmoothingCost(pyceres.CostFunction):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

    def Evaluate(self, parameters, residuals, jacobians):
        theta1 = parameters[0][0]
        theta2 = parameters[1][0]
        dt = parameters[2][0]
        if abs(dt) < 1e-9:
            return False

        delta_theta = normalize_angle(theta2 - theta1)

        omega = delta_theta / dt

        residuals[0] = self.weight * omega

        if jacobians is not None:
            inv_dt = 1.0 / dt
            w_inv_dt = self.weight * inv_dt

            if jacobians[0] is not None:
                jacobians[0][0] = -w_inv_dt

            if jacobians[1] is not None:
                jacobians[1][0] = w_inv_dt

            if jacobians[2] is not None:
                jacobians[2][0] = -residuals[0] * inv_dt

        return True


class SegmentTimeCost(pyceres.CostFunction):
    def __init__(self, weight: float):
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(self, parameters, residuals, jacobians):
        dt = parameters[0][0]

        w = self.weight
        residuals[:] = w * dt

        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = w

        return True


class TEBPlanner(TrajectoryPlanner):
    def __init__(
        self,
        start_pose: NDArray[np.floating] | list[float],
        goal_pose: NDArray[np.floating] | list[float],
        max_v: float,
        max_a: float,
        initial_step: float = 0.5,
    ):
        self.setup_poses(start_pose, goal_pose)
        self.max_v = max_v
        self.max_a = max_a
        self.initial_step = initial_step

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

    def _resize_trajectory(self, min_distance: float, max_distance: float): ...  # todo

    def move_start(
        self, new_xy, new_theta, min_distance: float, max_distance: float
    ) -> None:
        """
        Move the start point (and possibly remove/insert, based on distance to the next point).

        Args:
            new_xy: np.array of shape (2,) - new start position [x, y]
            new_theta: np.array of shape (1,) or scalar - new start heading
            min_distance: float - distance threshold to remove a point
            max_distance: float - distance threshold to create a point
        """
        if not self.optimization_xy:
            return

        new_xy = np.asarray(new_xy).reshape(2)
        new_theta = np.asarray(new_theta).reshape(1)

        if len(self.optimization_xy) <= 2:
            self.optimization_xy[0][:] = new_xy
            self.optimization_theta[0][:] = new_theta
            return

        distance = np.linalg.norm(new_xy - self.optimization_xy[1])

        if distance < min_distance:
            self.optimization_xy.pop(0)
            self.optimization_theta.pop(0)
            if self.optimization_dt:
                self.optimization_dt.pop(0)
            return

        self.optimization_xy[0][:] = new_xy
        self.optimization_theta[0][:] = new_theta

        if distance > max_distance:
            point1_xy = self.optimization_xy[1]
            self.optimization_xy.insert(
                1,
                np.array(
                    [(new_xy[0] + point1_xy[0]) / 2, (new_xy[1] + point1_xy[1]) / 2]
                ),
            )

            point1_theta = self.optimization_theta[1]
            diff = normalize_angle(point1_theta - new_theta)

            mean_theta = normalize_angle(new_theta.item() + diff.item() / 2.0)
            self.optimization_theta.insert(1, np.array([mean_theta], dtype=np.float64))

            half_dt = self.optimization_dt[0].item() / 2.0
            self.optimization_dt[0] = np.array([half_dt], dtype=np.float64)
            self.optimization_dt.insert(1, np.array([half_dt], dtype=np.float64))

    def move_goal(
        self, new_xy, new_theta, min_distance: float, max_distance: float
    ) -> None:
        """
        Move the goal point (and possibly remove/insert, based on distance to the previous point).
        """
        if not self.optimization_xy:
            return

        new_xy = np.asarray(new_xy).reshape(2)
        new_theta = np.asarray(new_theta).reshape(1)

        if len(self.optimization_xy) <= 2:
            self.optimization_xy[-1][:] = new_xy
            self.optimization_theta[-1][:] = new_theta
            return

        distance = np.linalg.norm(new_xy - self.optimization_xy[-2])

        if distance < min_distance:
            self.optimization_xy.pop(-1)
            self.optimization_theta.pop(-1)
            if self.optimization_dt:
                self.optimization_dt.pop(-1)

            self.optimization_xy[-1][:] = new_xy
            self.optimization_theta[-1][:] = new_theta
            return

        self.optimization_xy[-1][:] = new_xy
        self.optimization_theta[-1][:] = new_theta

        if distance > max_distance:
            prev_xy = self.optimization_xy[-2]
            mid_xy = np.array(
                [(new_xy[0] + prev_xy[0]) / 2, (new_xy[1] + prev_xy[1]) / 2]
            )

            prev_theta = self.optimization_theta[-2]
            diff = normalize_angle(prev_theta - new_theta)

            mean_theta = normalize_angle(new_theta.item() + diff.item() / 2.0)
            self.optimization_xy.insert(-1, mid_xy)
            self.optimization_theta.insert(-1, np.array([mean_theta], dtype=np.float64))

            half_dt = self.optimization_dt[-1].item() / 2.0
            self.optimization_dt[-1] = np.array([half_dt], dtype=np.float64)
            self.optimization_dt.insert(-1, np.array([half_dt], dtype=np.float64))

    def transform_trajectory(self, dx: float, dy: float, dtheta: float):
        """
        Transforms the internal trajectory guess to account for robot motion.
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
    def refine(self, iterations: int = 10) -> bool:
        if not self.optimization_xy:
            return False

        self._resize_trajectory(0.1, 2)  # todo: make into constants

        problem = pyceres.Problem()

        obstacle_cost = SegmentObstaclesCost(
            self.obstacles, weight=10.0, safety_radius=1.0
        )
        velocity_cost = SegmentVelocityCost(
            weight=10.0,
            max_v=self.max_v,
        )
        acceleration_cost = SegmentAccelerationCost(
            weight=10.0,
            max_a=self.max_a,
        )
        angular_velocity_cost = SegmentAngularVelocityCost(
            weight=5.0,
            max_omega=self.max_v / 2,
        )
        angular_acceleration_cost = SegmentAngularAccelerationCost(
            weight=10.0,
            max_alpha=self.max_a
            / 2,  # half linear acc as heuristic
        )
        kinematic_cost = SegmentKinematicsCost(
            weight=10.0,
        )
        heading_cost = SegmentHeadingCost(
            weight=10.0,
        )
        angular_smoothing_cost = SegmentAngularSmoothingCost(
            weight=0.5,
        )
        time_cost = SegmentTimeCost(
            weight=10.0,
        )

        n_points = len(self.optimization_xy)

        for i in range(n_points - 1):
            xy_curr = self.optimization_xy[i]
            xy_next = self.optimization_xy[i + 1]
            dt = self.optimization_dt[i]

            theta_curr = self.optimization_theta[i]
            theta_next = self.optimization_theta[i + 1]

            problem.add_residual_block(obstacle_cost, None, [xy_curr, xy_next])
            problem.add_residual_block(velocity_cost, None, [xy_curr, xy_next, dt])
            problem.add_residual_block(
                angular_velocity_cost, None, [theta_curr, theta_next, dt]
            )
            if i < n_points - 2:
                # problem.add_residual_block(
                #     acceleration_cost,
                #     None,
                #     [
                #         xy_curr,
                #         xy_next,
                #         self.optimization_xy[i + 2],
                #         dt,
                #         self.optimization_dt[i + 1],
                #     ],
                # )
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
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        return summary.termination_type == pyceres.TerminationType.CONVERGENCE
