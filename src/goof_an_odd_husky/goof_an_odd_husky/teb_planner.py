from goof_an_odd_husky.helpers import normalize_angle
from typing import override
from goof_an_odd_husky.trajectory_planner import (
    CircleObstacle,
    TrajectoryPlanner,
    LineObstacle,
)
import numpy as np
from numpy.typing import NDArray
import pyceres

DT_MIN = 0.01


def point_segment_distance(Px, Py, S1x, S1y, S2x, S2y):
    """Vectorized point-to-segment distance."""
    S1S2_x = S2x - S1x
    S1S2_y = S2y - S1y
    S1P_x = Px - S1x
    S1P_y = Py - S1y

    len_sq = S1S2_x**2 + S1S2_y**2
    len_sq = np.maximum(len_sq, 1e-10)

    t = (S1P_x * S1S2_x + S1P_y * S1S2_y) / len_sq
    t = np.clip(t, 0.0, 1.0)

    Proj_x = S1x + t * S1S2_x
    Proj_y = S1y + t * S1S2_y

    u_x = Px - Proj_x
    u_y = Py - Proj_y

    d = np.sqrt(u_x**2 + u_y**2 + 1e-10)
    return d, u_x, u_y, t


class SegmentLineObstaclesCost(pyceres.CostFunction):
    def __init__(
        self,
        line_obstacles: list[LineObstacle],
        weight: float,
        safety_radius: float,
    ):
        super().__init__()

        self.C_x = np.array([obs.x1 for obs in line_obstacles], dtype=np.float64)
        self.C_y = np.array([obs.y1 for obs in line_obstacles], dtype=np.float64)
        self.D_x = np.array([obs.x2 for obs in line_obstacles], dtype=np.float64)
        self.D_y = np.array([obs.y2 for obs in line_obstacles], dtype=np.float64)

        self.n_obstacles = len(self.C_x)
        self.weight = weight
        self.safety_radius = safety_radius

        self.set_num_residuals(self.n_obstacles)
        self.set_parameter_block_sizes([2, 2])

    def Evaluate(self, parameters, residuals, jacobians):
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
        min_idx = np.argmin(all_d, axis=0)
        d_min = all_d[min_idx, np.arange(self.n_obstacles)]

        CD_x, CD_y = self.D_x - self.C_x, self.D_y - self.C_y
        CA_x, CA_y = A_x - self.C_x, A_y - self.C_y
        CB_x, CB_y = B_x - self.C_x, B_y - self.C_y

        cp1 = CD_x * CA_y - CD_y * CA_x
        cp2 = CD_x * CB_y - CD_y * CB_x
        diff_side_CD = (cp1 * cp2) <= 0.0

        AB_x, AB_y = B_x - A_x, B_y - A_y
        AC_x, AC_y = self.C_x - A_x, self.C_y - A_y
        AD_x, AD_y = self.D_x - A_x, self.D_y - A_y

        cp3 = AB_x * AC_y - AB_y * AC_x
        cp4 = AB_x * AD_y - AB_y * AD_x
        diff_side_AB = (cp3 * cp4) <= 0.0

        intersect = diff_side_CD & diff_side_AB

        sign_mask = np.where(intersect, -1.0, 1.0)
        errors = self.safety_radius - sign_mask * d_min

        active_mask = errors > 0.0
        residuals[:] = np.where(active_mask, self.weight * errors, 0.0)

        if jacobians is not None:
            inv_d1 = 1.0 / d1
            inv_d2 = 1.0 / d2
            inv_d3 = 1.0 / d3
            inv_d4 = 1.0 / d4

            z = np.zeros(self.n_obstacles)

            gA1_x, gA1_y = u1_x * inv_d1, u1_y * inv_d1
            gB1_x, gB1_y = z, z

            gA2_x, gA2_y = z, z
            gB2_x, gB2_y = u2_x * inv_d2, u2_y * inv_d2

            gA3_x, gA3_y = -(1.0 - t3) * u3_x * inv_d3, -(1.0 - t3) * u3_y * inv_d3
            gB3_x, gB3_y = -t3 * u3_x * inv_d3, -t3 * u3_y * inv_d3

            gA4_x, gA4_y = -(1.0 - t4) * u4_x * inv_d4, -(1.0 - t4) * u4_y * inv_d4
            gB4_x, gB4_y = -t4 * u4_x * inv_d4, -t4 * u4_y * inv_d4

            gA_x_min = np.choose(min_idx, [gA1_x, gA2_x, gA3_x, gA4_x])
            gA_y_min = np.choose(min_idx, [gA1_y, gA2_y, gA3_y, gA4_y])
            gB_x_min = np.choose(min_idx, [gB1_x, gB2_x, gB3_x, gB4_x])
            gB_y_min = np.choose(min_idx, [gB1_y, gB2_y, gB3_y, gB4_y])

            j_scaler = np.where(active_mask, self.weight * (-sign_mask), 0.0)

            if jacobians[0] is not None:
                J_A_x = j_scaler * gA_x_min
                J_A_y = j_scaler * gA_y_min
                jacobians[0][:] = np.vstack((J_A_x, J_A_y)).T.ravel()

            if jacobians[1] is not None:
                J_B_x = j_scaler * gB_x_min
                J_B_y = j_scaler * gB_y_min
                jacobians[1][:] = np.vstack((J_B_x, J_B_y)).T.ravel()

        return True


class SegmentCircleObstaclesCost(pyceres.CostFunction):
    def __init__(
        self,
        circle_obstacles: list[CircleObstacle],
        weight: float,
        safety_radius: float,
    ):
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


class StartAccelerationCost(pyceres.CostFunction):
    def __init__(self, weight: float, max_a: float, current_v: tuple[float, float]):
        super().__init__()
        self.weight = weight
        self.max_a = max_a
        self.vx_start = current_v[0]
        self.vy_start = current_v[1]

        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

    def Evaluate(self, parameters, residuals, jacobians):
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
    def __init__(self, weight: float, max_alpha: float, current_omega: float):
        super().__init__()
        self.weight = weight
        self.max_alpha = max_alpha
        self.omega_start = current_omega
        self.set_parameter_block_sizes([1, 1, 1])
        self.set_num_residuals(1)

    def Evaluate(self, parameters, residuals, jacobians):
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
        initial_step: float = 1.0,
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

        self.resize_trajectory(0.15, 3)  # todo: make into constants

        if len(self.optimization_xy) <= 2:
            return False

        problem = pyceres.Problem()

        circle_obstacles = [
            obs for obs in self.obstacles if isinstance(obs, CircleObstacle)
        ]
        line_obstacles = [
            obs for obs in self.obstacles if isinstance(obs, LineObstacle)
        ]

        circle_cost_func = None
        if circle_obstacles:
            circle_cost_func = SegmentCircleObstaclesCost(
                circle_obstacles, weight=10.0, safety_radius=1.0
            )

        line_cost_func = None
        if line_obstacles:
            line_cost_func = SegmentLineObstaclesCost(
                line_obstacles, weight=10.0, safety_radius=1.0
            )

        start_theta = self.optimization_theta[0]
        start_vx = current_velocity * np.cos(start_theta).item()
        start_vy = current_velocity * np.sin(start_theta).item()
        current_velocity_vec = (start_vx, start_vy)

        velocity_cost = SegmentVelocityCost(weight=10.0, max_v=self.max_v)
        angular_velocity_cost = SegmentAngularVelocityCost(
            weight=5.0, max_omega=self.max_v / 2
        )
        acceleration_cost = SegmentAccelerationCost(weight=10.0, max_a=self.max_a)
        angular_acceleration_cost = SegmentAngularAccelerationCost(
            weight=10.0, max_alpha=self.max_a / 2
        )
        start_acceleration_cost = StartAccelerationCost(
            weight=10.0, max_a=self.max_a, current_v=current_velocity_vec
        )
        start_angular_acceleration_cost = StartAngularAccelerationCost(
            weight=10.0, max_alpha=self.max_a, current_omega=current_omega
        )
        kinematic_cost = SegmentKinematicsCost(weight=10.0)
        heading_cost = SegmentHeadingCost(weight=10.0)
        angular_smoothing_cost = SegmentAngularSmoothingCost(weight=0.5)
        time_cost = SegmentTimeCost(weight=10.0)

        n_points = len(self.optimization_xy)

        for i in range(n_points - 1):
            xy_curr = self.optimization_xy[i]
            xy_next = self.optimization_xy[i + 1]
            dt = self.optimization_dt[i]

            theta_curr = self.optimization_theta[i]
            theta_next = self.optimization_theta[i + 1]

            if circle_cost_func is not None:
                problem.add_residual_block(circle_cost_func, None, [xy_curr, xy_next])

            if line_cost_func is not None:
                problem.add_residual_block(line_cost_func, None, [xy_curr, xy_next])

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
        options.linear_solver_type = pyceres.LinearSolverType.DENSE_QR
        options.minimizer_progress_to_stdout = False

        summary = pyceres.SolverSummary()
        pyceres.solve(options, problem, summary)

        return summary.termination_type == pyceres.TerminationType.CONVERGENCE
