from goof_an_odd_husky_common.helpers import (
    normalize_angle,
    point_segment_distance,
)
from goof_an_odd_husky_common.obstacles import CircleObstacle, LineObstacle

import goof_costs

import numpy as np
from numpy.typing import NDArray
import pyceres


class _CppCostWrapper(pyceres.CostFunction):
    def _eval_cpp(self, parameters, residuals, jacobians):
        param_ptrs = [p.__array_interface__["data"][0] for p in parameters]
        param_array = np.array(param_ptrs, dtype=np.uint64)

        res_ptr = residuals.__array_interface__["data"][0]

        jac_ptr = 0
        if jacobians is not None:
            jac_ptrs = [
                j.__array_interface__["data"][0] if j is not None else 0
                for j in jacobians
            ]
            jac_array = np.array(jac_ptrs, dtype=np.uint64)
            jac_ptr = jac_array.__array_interface__["data"][0]

        return self.cpp_impl.evaluate(
            len(parameters),
            param_array.__array_interface__["data"][0],
            res_ptr,
            jac_ptr,
        )


class SegmentLineObstaclesCost(_CppCostWrapper):
    def __init__(self, line_obstacles, weight, safety_radius, softmin_alpha=-7.0):
        super().__init__()

        self.C_x = np.array([o.x1 for o in line_obstacles], dtype=np.float64)
        self.C_y = np.array([o.y1 for o in line_obstacles], dtype=np.float64)
        self.D_x = np.array([o.x2 for o in line_obstacles], dtype=np.float64)
        self.D_y = np.array([o.y2 for o in line_obstacles], dtype=np.float64)

        n = len(self.C_x)

        self.set_num_residuals(n)
        self.set_parameter_block_sizes([2, 2])

        self.cpp_impl = goof_costs.SegmentLineObstaclesCostExt(
            self.C_x.tolist(),
            self.C_y.tolist(),
            self.D_x.tolist(),
            self.D_y.tolist(),
            weight,
            safety_radius,
            softmin_alpha,
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentCircleObstaclesCost(_CppCostWrapper):
    def __init__(self, circle_obstacles, weight: float, safety_radius: float):
        super().__init__()

        self.obstacles_x = np.array([o.x for o in circle_obstacles], dtype=np.float64)
        self.obstacles_y = np.array([o.y for o in circle_obstacles], dtype=np.float64)
        self.obstacles_r = np.array([o.radius for o in circle_obstacles], dtype=np.float64)

        n = len(self.obstacles_r)

        self.set_num_residuals(n)
        self.set_parameter_block_sizes([2, 2])

        self.cpp_impl = goof_costs.SegmentCircleObstaclesCostExt(
            self.obstacles_x.tolist(),
            self.obstacles_y.tolist(),
            self.obstacles_r.tolist(),
            weight,
            safety_radius,
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentVelocityCost(_CppCostWrapper):
    def __init__(self, weight: float, max_v: float):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

        self.cpp_impl = goof_costs.SegmentVelocityCostExt(weight, max_v)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentAngularVelocityCost(_CppCostWrapper):
    def __init__(self, weight: float, max_omega: float):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

        self.cpp_impl = goof_costs.SegmentAngularVelocityCostExt(
            weight, max_omega
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentAccelerationCost(_CppCostWrapper):
    def __init__(self, weight: float, max_a: float):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 2, 1, 1])

        self.cpp_impl = goof_costs.SegmentAccelerationCostExt(weight, max_a)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentAngularAccelerationCost(_CppCostWrapper):
    def __init__(self, weight: float, max_alpha: float):
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1, 1, 1])

        self.cpp_impl = goof_costs.SegmentAngularAccelerationCostExt(
            weight, max_alpha
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class StartAccelerationCost(_CppCostWrapper):
    def __init__(self, weight: float, max_a: float, current_v: tuple[float, float]) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 2, 1])

        self.cpp_impl = goof_costs.StartAccelerationCostExt(
            weight, max_a, current_v,
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class StartAngularAccelerationCost(_CppCostWrapper):
    def __init__(self, weight: float, max_alpha: float, current_omega: float) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

        self.cpp_impl = goof_costs.StartAngularAccelerationCostExt(
            weight, max_alpha, current_omega
        )

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentKinematicsCost(_CppCostWrapper):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2, 1])

        self.cpp_impl = goof_costs.SegmentKinematicsCostExt(weight)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentHeadingCost(_CppCostWrapper):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([2, 1, 2])

        self.cpp_impl = goof_costs.SegmentHeadingCostExt(weight)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentAngularSmoothingCost(_CppCostWrapper):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1, 1, 1])

        self.cpp_impl = goof_costs.SegmentAngularSmoothingCostExt(weight)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)


class SegmentTimeCost(_CppCostWrapper):
    def __init__(self, weight: float) -> None:
        super().__init__()
        self.set_num_residuals(1)
        self.set_parameter_block_sizes([1])

        self.cpp_impl = goof_costs.SegmentTimeCostExt(weight)

    def Evaluate(self, parameters, residuals, jacobians):
        return self._eval_cpp(parameters, residuals, jacobians)
