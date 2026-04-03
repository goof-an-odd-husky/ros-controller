from goof_an_odd_husky_common.helpers import (
    normalize_angle,
    point_segment_distance,
    segments_intersect,
)
from goof_an_odd_husky_common.obstacles import CircleObstacle, LineObstacle

import numpy as np
from numpy.typing import NDArray
import pyceres


class SegmentLineObstaclesCost(pyceres.CostFunction):
    """Ceres cost function penalizing proximity to line obstacles.

    Attributes:
        C_x: Numpy array of line obstacle start X coordinates.
        C_y: Numpy array of line obstacle start Y coordinates.
        D_x: Numpy array of line obstacle end X coordinates.
        D_y: Numpy array of line obstacle end Y coordinates.
        n_obstacles: Number of line obstacles.
        weight: The penalty weight multiplier.
        safety_radius: Minimum allowed distance to an obstacle.
        softmin_alpha: Parameter controlling the smoothness of the min function.
    """

    C_x: NDArray[np.float64]
    C_y: NDArray[np.float64]
    D_x: NDArray[np.float64]
    D_y: NDArray[np.float64]
    n_obstacles: int
    weight: float
    safety_radius: float
    softmin_alpha: float

    def __init__(
        self,
        line_obstacles: list[LineObstacle],
        weight: float,
        safety_radius: float,
        softmin_alpha: float = -7.0,
    ) -> None:
        """Initialize the SegmentLineObstaclesCost.

        Args:
            line_obstacles: List of detected line obstacles.
            weight: Cost multiplier.
            safety_radius: Allowed distance before penalty applies.
            softmin_alpha: Softmin smoothing factor.
        """
        super().__init__()

        self.C_x = np.array([obs.x1 for obs in line_obstacles], dtype=np.float64)
        self.C_y = np.array([obs.y1 for obs in line_obstacles], dtype=np.float64)
        self.D_x = np.array([obs.x2 for obs in line_obstacles], dtype=np.float64)
        self.D_y = np.array([obs.y2 for obs in line_obstacles], dtype=np.float64)

        self.n_obstacles = len(self.C_x)
        self.weight = weight
        self.safety_radius = safety_radius
        self.softmin_alpha = softmin_alpha

        self.set_num_residuals(self.n_obstacles)
        self.set_parameter_block_sizes([2, 2])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_xy, PointB_xy].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]

        d1, u1_x, u1_y, _ = point_segment_distance(
            A_x, A_y, self.C_x, self.C_y, self.D_x, self.D_y
        )
        d2, u2_x, u2_y, _ = point_segment_distance(
            B_x, B_y, self.C_x, self.C_y, self.D_x, self.D_y
        )
        d3, u3_x, u3_y, t3 = point_segment_distance(
            self.C_x, self.C_y, A_x, A_y, B_x, B_y
        )
        d4, u4_x, u4_y, t4 = point_segment_distance(
            self.D_x, self.D_y, A_x, A_y, B_x, B_y
        )

        all_d = np.vstack([d1, d2, d3, d4])
        scaled_d = self.softmin_alpha * all_d
        max_scaled_d = np.max(scaled_d, axis=0)
        exp_d = np.exp(scaled_d - max_scaled_d)
        sum_exp = np.sum(exp_d, axis=0)
        d_min = (np.log(sum_exp) + max_scaled_d) / self.softmin_alpha

        weights = exp_d / sum_exp
        w1, w2, w3, w4 = weights[0], weights[1], weights[2], weights[3]

        errors = self.safety_radius - d_min
        active_mask = errors > 0.0
        residuals[:] = np.where(active_mask, self.weight * errors, 0.0)

        if jacobians is not None:
            inv_d1 = 1.0 / np.maximum(d1, 1e-8)
            inv_d2 = 1.0 / np.maximum(d2, 1e-8)
            inv_d3 = 1.0 / np.maximum(d3, 1e-8)
            inv_d4 = 1.0 / np.maximum(d4, 1e-8)

            gA_x_min = (
                w1 * u1_x * inv_d1
                + w3 * (-(1.0 - t3) * u3_x * inv_d3)
                + w4 * (-(1.0 - t4) * u4_x * inv_d4)
            )
            gA_y_min = (
                w1 * u1_y * inv_d1
                + w3 * (-(1.0 - t3) * u3_y * inv_d3)
                + w4 * (-(1.0 - t4) * u4_y * inv_d4)
            )
            gB_x_min = (
                w2 * u2_x * inv_d2
                + w3 * (-t3 * u3_x * inv_d3)
                + w4 * (-t4 * u4_x * inv_d4)
            )
            gB_y_min = (
                w2 * u2_y * inv_d2
                + w3 * (-t3 * u3_y * inv_d3)
                + w4 * (-t4 * u4_y * inv_d4)
            )

            j_scaler = np.where(active_mask, -self.weight, 0.0)

            if jacobians[0] is not None:
                jacobians[0][:] = np.vstack(
                    (j_scaler * gA_x_min, j_scaler * gA_y_min)
                ).T.ravel()
            if jacobians[1] is not None:
                jacobians[1][:] = np.vstack(
                    (j_scaler * gB_x_min, j_scaler * gB_y_min)
                ).T.ravel()

        return True
        return True


class SegmentCircleObstaclesCost(pyceres.CostFunction):
    """Ceres cost function penalizing proximity to circular obstacles.

    Attributes:
        obstacles_x: Numpy array of obstacle X coordinates.
        obstacles_y: Numpy array of obstacle Y coordinates.
        obstacles_r: Numpy array of obstacle radii.
        n_obstacles: Number of obstacles.
        weight: Penalty weight multiplier.
        safety_radius: Added buffer radius to the obstacle.
    """

    obstacles_x: NDArray[np.float64]
    obstacles_y: NDArray[np.float64]
    obstacles_r: NDArray[np.float64]
    n_obstacles: int
    weight: float
    safety_radius: float

    def __init__(
        self,
        circle_obstacles: list[CircleObstacle],
        weight: float,
        safety_radius: float,
    ) -> None:
        """Initialize the SegmentCircleObstaclesCost.

        Args:
            circle_obstacles: List of detected circular obstacles.
            weight: Cost multiplier.
            safety_radius: Buffer distance added to the obstacle radius.
        """
        super().__init__()

        self.obstacles_x = np.array(
            [obs.x for obs in circle_obstacles], dtype=np.float64
        )
        self.obstacles_y = np.array(
            [obs.y for obs in circle_obstacles], dtype=np.float64
        )
        self.obstacles_r = np.array(
            [obs.radius for obs in circle_obstacles], dtype=np.float64
        )

        self.n_obstacles = len(self.obstacles_r)
        self.weight = weight
        self.safety_radius = safety_radius

        self.set_num_residuals(self.n_obstacles)
        self.set_parameter_block_sizes([2, 2])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_xy, PointB_xy].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        A_x, A_y = parameters[0]
        B_x, B_y = parameters[1]

        AB_x = B_x - A_x
        AB_y = B_y - A_y
        AB_len_sq = max(AB_x**2 + AB_y**2, 1e-10)

        AO_x = self.obstacles_x - A_x
        AO_y = self.obstacles_y - A_y

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
            inv_dist = 1.0 / O1O_len
            d_hat_x = O1O_x * inv_dist
            d_hat_y = O1O_y * inv_dist

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
    """Ceres cost function penalizing velocities exceeding the maximum.

    Attributes:
        weight: Penalty weight multiplier.
        max_v: Maximum allowed linear velocity.
    """

    weight: float
    max_v: float

    def __init__(self, weight: float, max_v: float) -> None:
        """Initialize SegmentVelocityCost.

        Args:
            weight: Cost multiplier.
            max_v: Maximum linear velocity allowed.
        """
        super().__init__()
        self.weight = weight
        self.max_v = max_v

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_xy, PointB_xy, dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
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

        diff = 0.0
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
                dv_dt = -v * inv_dt
                jacobians[2][0] = dr_dv * dv_dt

        return True


class SegmentAngularVelocityCost(pyceres.CostFunction):
    """Ceres cost function penalizing angular velocities exceeding the maximum.

    Attributes:
        weight: Penalty weight multiplier.
        max_omega: Maximum allowed angular velocity.
    """

    weight: float
    max_omega: float

    def __init__(self, weight: float, max_omega: float) -> None:
        """Initialize SegmentAngularVelocityCost.

        Args:
            weight: Cost multiplier.
            max_omega: Maximum angular velocity allowed.
        """
        super().__init__()
        self.weight = weight
        self.max_omega = max_omega

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_theta, PointB_theta, dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        A_theta = parameters[0][0]
        B_theta = parameters[1][0]
        dt = parameters[2][0]
        if abs(dt) < 1e-9:
            return False
        inv_dt = 1 / dt

        delta_theta = normalize_angle(B_theta - A_theta)

        omega = delta_theta * inv_dt

        diff = 0.0
        if omega > self.max_omega:
            diff = omega - self.max_omega
        elif omega < -self.max_omega:
            diff = omega + self.max_omega

        w = self.weight
        residuals[0] = w * diff

        if jacobians is not None:
            if abs(omega) <= self.max_omega:
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
    """Ceres cost function penalizing linear accelerations exceeding the maximum.

    Attributes:
        weight: Penalty weight multiplier.
        max_a: Maximum allowed linear acceleration.
    """

    weight: float
    max_a: float

    def __init__(self, weight: float, max_a: float) -> None:
        """Initialize SegmentAccelerationCost.

        Args:
            weight: Cost multiplier.
            max_a: Maximum linear acceleration allowed.
        """
        super().__init__()
        self.weight = weight
        self.max_a = max_a
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 2, 1, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_xy, PointB_xy, PointC_xy, dt1, dt2].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        A = parameters[0]
        B = parameters[1]
        C = parameters[2]
        dt1 = parameters[3][0]
        dt2 = parameters[4][0]

        if dt1 < 1e-5 or dt2 < 1e-5:
            dt1 = max(dt1, 1e-5)
            dt2 = max(dt2, 1e-5)

        vx1 = (B[0] - A[0]) / dt1
        vy1 = (B[1] - A[1]) / dt1

        vx2 = (C[0] - B[0]) / dt2
        vy2 = (C[1] - B[1]) / dt2

        dt_avg = (dt1 + dt2) / 2.0
        inv_dt_avg = 1.0 / dt_avg

        ax = (vx2 - vx1) * inv_dt_avg
        ay = (vy2 - vy1) * inv_dt_avg

        a_sq = ax**2 + ay**2
        a_norm = np.sqrt(a_sq + 1e-12)

        diff = a_norm - self.max_a
        if diff <= 0:
            residuals[0] = 0.0
            if jacobians is not None:
                for i in range(5):
                    if jacobians[i] is not None:
                        jacobians[i][:] = 0.0
            return True

        residuals[0] = self.weight * diff

        if jacobians is not None:
            w_norm = self.weight / a_norm

            dax_dA = inv_dt_avg / dt1
            jac_A = [w_norm * (ax * dax_dA), w_norm * (ay * dax_dA)]

            dax_dC = inv_dt_avg / dt2
            jac_C = [w_norm * (ax * dax_dC), w_norm * (ay * dax_dC)]

            dax_dB = inv_dt_avg * (-1.0 / dt2 - 1.0 / dt1)
            jac_B = [w_norm * (ax * dax_dB), w_norm * (ay * dax_dB)]

            d_ax_dt1 = -ax / (2 * dt_avg) + inv_dt_avg * (vx1 / dt1)
            d_ay_dt1 = -ay / (2 * dt_avg) + inv_dt_avg * (vy1 / dt1)
            jac_dt1 = w_norm * (ax * d_ax_dt1 + ay * d_ay_dt1)

            d_ax_dt2 = -ax / (2 * dt_avg) - inv_dt_avg * (vx2 / dt2)
            d_ay_dt2 = -ay / (2 * dt_avg) - inv_dt_avg * (vy2 / dt2)
            jac_dt2 = w_norm * (ax * d_ax_dt2 + ay * d_ay_dt2)

            if jacobians[0] is not None:
                jacobians[0][:] = jac_A
            if jacobians[1] is not None:
                jacobians[1][:] = jac_B
            if jacobians[2] is not None:
                jacobians[2][:] = jac_C
            if jacobians[3] is not None:
                jacobians[3][:] = [jac_dt1]
            if jacobians[4] is not None:
                jacobians[4][:] = [jac_dt2]

        return True


class SegmentAngularAccelerationCost(pyceres.CostFunction):
    """Ceres cost function penalizing angular accelerations exceeding the maximum.

    Attributes:
        weight: Penalty weight multiplier.
        max_alpha: Maximum allowed angular acceleration.
    """

    weight: float
    max_alpha: float

    def __init__(self, weight: float, max_alpha: float) -> None:
        """Initialize SegmentAngularAccelerationCost.

        Args:
            weight: Cost multiplier.
            max_alpha: Maximum angular acceleration allowed.
        """
        super().__init__()
        self.weight = weight
        self.max_alpha = max_alpha

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1, 1, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [PointA_theta, PointB_theta, PointC_theta, dt1, dt2].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
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


class StartAccelerationCost(pyceres.CostFunction):
    """Ceres cost function penalizing starting linear acceleration.

    Attributes:
        weight: Penalty weight multiplier.
        max_a: Maximum allowed linear acceleration.
        vx_start: Initial X velocity.
        vy_start: Initial Y velocity.
    """

    weight: float
    max_a: float
    vx_start: float
    vy_start: float

    def __init__(
        self, weight: float, max_a: float, current_v: tuple[float, float]
    ) -> None:
        """Initialize StartAccelerationCost.

        Args:
            weight: Cost multiplier.
            max_a: Maximum linear acceleration allowed.
            current_v: The starting linear velocity vector (vx, vy).
        """
        super().__init__()
        self.weight = weight
        self.max_a = max_a
        self.vx_start = current_v[0]
        self.vy_start = current_v[1]

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [Point_start, Point_next, dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        P_start = parameters[0]
        P_next = parameters[1]
        dt = parameters[2][0]

        if dt < 1e-5:
            dt = 1e-5

        vx_next = (P_next[0] - P_start[0]) / dt
        vy_next = (P_next[1] - P_start[1]) / dt

        ax = (vx_next - self.vx_start) / dt
        ay = (vy_next - self.vy_start) / dt

        a_norm = np.sqrt(ax**2 + ay**2 + 1e-12)
        diff = a_norm - self.max_a

        if diff <= 0:
            residuals[0] = 0.0
            if jacobians:
                if jacobians[0] is not None:
                    jacobians[0][:] = 0.0
                if jacobians[1] is not None:
                    jacobians[1][:] = 0.0
                if jacobians[2] is not None:
                    jacobians[2][:] = 0.0
            return True

        residuals[0] = self.weight * diff

        if jacobians:
            w_norm = self.weight / a_norm
            inv_dt = 1.0 / dt
            inv_dt_sq = inv_dt * inv_dt

            if jacobians[0] is not None:
                jacobians[0][:] = 0.0
            if jacobians[1] is not None:
                jacobians[1][0] = w_norm * (ax * inv_dt_sq)
                jacobians[1][1] = w_norm * (ay * inv_dt_sq)
            if jacobians[2] is not None:
                d_ax_dt = (-2 * vx_next + self.vx_start) * (inv_dt**2)
                d_ay_dt = (-2 * vy_next + self.vy_start) * (inv_dt**2)
                jacobians[2][:] = [w_norm * (ax * d_ax_dt + ay * d_ay_dt)]

        return True


class StartAngularAccelerationCost(pyceres.CostFunction):
    """Ceres cost function penalizing starting angular acceleration.

    Attributes:
        weight: Penalty weight multiplier.
        max_alpha: Maximum allowed angular acceleration.
        omega_start: Initial angular velocity.
    """

    weight: float
    max_alpha: float
    omega_start: float

    def __init__(self, weight: float, max_alpha: float, current_omega: float) -> None:
        """Initialize StartAngularAccelerationCost.

        Args:
            weight: Cost multiplier.
            max_alpha: Maximum angular acceleration allowed.
            current_omega: The starting angular velocity.
        """
        super().__init__()
        self.weight = weight
        self.max_alpha = max_alpha
        self.omega_start = current_omega
        self.set_parameter_block_sizes([1, 1, 1])
        self.set_num_residuals(1)

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [theta_start, theta_next, dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        th_start = parameters[0][0]
        th_next = parameters[1][0]
        dt = parameters[2][0]
        if dt < 1e-5:
            dt = 1e-5

        delta_th = th_next - th_start
        delta_th = (delta_th + np.pi) % (2 * np.pi) - np.pi

        omega_next = delta_th / dt
        alpha = (omega_next - self.omega_start) / dt

        abs_alpha = abs(alpha)
        diff = abs_alpha - self.max_alpha

        if diff <= 0:
            residuals[0] = 0.0
            if jacobians:
                for j in jacobians:
                    if j is not None:
                        j[:] = 0.0
            return True

        sign = 1.0 if alpha > 0 else -1.0
        residuals[0] = self.weight * diff

        if jacobians:
            w = self.weight * sign
            inv_dt = 1.0 / dt
            inv_dt2 = inv_dt * inv_dt

            d_alpha_dth = inv_dt2

            d_alpha_dt = -2 * delta_th * inv_dt * inv_dt2 + self.omega_start * inv_dt2

            if jacobians[0] is not None:
                jacobians[0][:] = 0.0
            if jacobians[1] is not None:
                jacobians[1][:] = [w * d_alpha_dth]
            if jacobians[2] is not None:
                jacobians[2][:] = [w * d_alpha_dt]

        return True


class SegmentKinematicsCost(pyceres.CostFunction):
    """Ceres cost function enforcing non-holonomic kinematic constraints.

    Attributes:
        weight: Penalty weight multiplier.
    """

    weight: float

    def __init__(self, weight: float) -> None:
        """Initialize SegmentKinematicsCost.

        Args:
            weight: Cost multiplier.
        """
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [xy1, theta1, xy2, theta2].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
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
    """Ceres cost function penalizing driving backwards.

    Attributes:
        weight: Penalty weight multiplier.
    """

    weight: float

    def __init__(self, weight: float) -> None:
        """Initialize SegmentHeadingCost.

        Args:
            weight: Cost multiplier.
        """
        super().__init__()
        self.weight = weight
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [xy1, theta1, xy2].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
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
    """Ceres cost function minimizing angular velocity changes to smooth the trajectory.

    Attributes:
        weight: Penalty weight multiplier.
    """

    weight: float

    def __init__(self, weight: float) -> None:
        """Initialize SegmentAngularSmoothingCost.

        Args:
            weight: Cost multiplier.
        """
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [theta1, theta2, dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
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
    """Ceres cost function penalizing traversal time to encourage faster completion.

    Attributes:
        weight: Penalty weight multiplier.
    """

    weight: float

    def __init__(self, weight: float) -> None:
        """Initialize SegmentTimeCost.

        Args:
            weight: Cost multiplier.
        """
        super().__init__()
        self.weight = weight

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

    def Evaluate(
        self,
        parameters: list[NDArray[np.float64]],
        residuals: NDArray[np.float64],
        jacobians: list[NDArray[np.float64] | None] | None,
    ) -> bool:
        """Evaluate the cost function and its jacobians.

        Args:
            parameters: List containing [dt].
            residuals: Output array for the calculated residuals.
            jacobians: Optional output array for the calculated jacobians.

        Returns:
            bool: True indicating successful evaluation.
        """
        dt = parameters[0][0]

        w = self.weight
        residuals[:] = w * dt

        if jacobians is not None and jacobians[0] is not None:
            jacobians[0][0] = w

        return True
