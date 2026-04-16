#include <ceres/ceres.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

constexpr double kPi = 3.141592653589793238462643383279502884;
constexpr double kTwoPi = 2.0 * kPi;

inline double normalize_angle_double(double angle) {
  angle = std::fmod(angle + kPi, kTwoPi);
  if (angle < 0.0) {
    angle += kTwoPi;
  }
  return angle - kPi;
}

template <typename T>
inline void point_segment_distance(const T Px, const T Py, const T S1x,
                                   const T S1y, const T S2x, const T S2y,
                                   T* d, T* u_x, T* u_y, T* t) {
  const T S1S2_x = S2x - S1x;
  const T S1S2_y = S2y - S1y;
  const T S1P_x = Px - S1x;
  const T S1P_y = Py - S1y;

  T len_sq = S1S2_x * S1S2_x + S1S2_y * S1S2_y;
  len_sq = ceres::fmax(len_sq, T(1e-10));

  T tt = (S1P_x * S1S2_x + S1P_y * S1S2_y) / len_sq;
  tt = ceres::fmin(T(1.0), ceres::fmax(T(0.0), tt));

  const T Proj_x = S1x + tt * S1S2_x;
  const T Proj_y = S1y + tt * S1S2_y;

  const T ux = Px - Proj_x;
  const T uy = Py - Proj_y;

  const T dist = ceres::sqrt(ux * ux + uy * uy + T(1e-10));

  *d = dist;
  *u_x = ux;
  *u_y = uy;
  *t = tt;
}

inline void zero_block(double* block, int size) {
  if (block == nullptr) {
    return;
  }
  for (int i = 0; i < size; ++i) {
    block[i] = 0.0;
  }
}

inline void evaluate_cost(ceres::CostFunction* cost_func,
                          std::size_t /*n_params*/, std::uintptr_t parameters_ptr,
                          std::uintptr_t residuals_ptr,
                          std::uintptr_t jacobians_ptr) {
  const double* const* params =
      reinterpret_cast<const double* const*>(parameters_ptr);
  double* residuals = reinterpret_cast<double*>(residuals_ptr);
  double** jacobians =
      jacobians_ptr == 0 ? nullptr : reinterpret_cast<double**>(jacobians_ptr);

  bool success;
  {
    py::gil_scoped_release release;
    success = cost_func->Evaluate(params, residuals, jacobians);
  }
  (void)success;
}

class SegmentVelocityCost : public ceres::SizedCostFunction<1, 2, 2, 1> {
public:
  SegmentVelocityCost(double weight, double max_v)
      : weight_(weight), max_v_(max_v) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double* A = parameters[0];
    const double* B = parameters[1];
    double dt = parameters[2][0];

    if (std::abs(dt) < 1e-9) {
      return false;
    }

    const double inv_dt = 1.0 / dt;
    const double AB_x = B[0] - A[0];
    const double AB_y = B[1] - A[1];
    const double AB_len = std::sqrt(AB_x * AB_x + AB_y * AB_y + 1e-10);
    const double v = AB_len * inv_dt;

    double diff = 0.0;
    if (v > max_v_) {
      diff = v - max_v_;
    }

    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      if (v <= max_v_) {
        zero_block(jacobians[0], 2);
        zero_block(jacobians[1], 2);
        zero_block(jacobians[2], 1);
        return true;
      }

      const double dr_dv = weight_;
      const double scale = dr_dv / (AB_len * dt);

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = -AB_x * scale;
        jacobians[0][1] = -AB_y * scale;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = AB_x * scale;
        jacobians[1][1] = AB_y * scale;
      }
      if (jacobians[2] != nullptr) {
        const double dv_dt = -v * inv_dt;
        jacobians[2][0] = dr_dv * dv_dt;
      }
    }

    return true;
  }

private:
  double weight_;
  double max_v_;
};

class SegmentAngularVelocityCost : public ceres::SizedCostFunction<1, 1, 1, 1> {
public:
  SegmentAngularVelocityCost(double weight, double max_omega)
      : weight_(weight), max_omega_(max_omega) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double th_start = parameters[0][0];
    const double th_next = parameters[1][0];
    double dt = parameters[2][0];

    if (std::abs(dt) < 1e-9) {
      return false;
    }

    const double inv_dt = 1.0 / dt;
    const double delta_th = normalize_angle_double(th_next - th_start);
    const double omega = delta_th * inv_dt;

    double diff = 0.0;
    if (omega > max_omega_) {
      diff = omega - max_omega_;
    } else if (omega < -max_omega_) {
      diff = omega + max_omega_;
    }

    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      if (std::abs(omega) <= max_omega_) {
        zero_block(jacobians[0], 1);
        zero_block(jacobians[1], 1);
        zero_block(jacobians[2], 1);
        return true;
      }

      const double dr_ddelta_th = weight_ * inv_dt;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = -dr_ddelta_th;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = dr_ddelta_th;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = -weight_ * omega * inv_dt;
      }
    }

    return true;
  }

private:
  double weight_;
  double max_omega_;
};

class StartAccelerationCost : public ceres::SizedCostFunction<1, 2, 2, 1> {
public:
  StartAccelerationCost(double weight, double max_a, std::pair<double, double> current_v)
      : weight_(weight), max_a_(max_a), vx_start_(current_v.first),
        vy_start_(current_v.second) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double* P_start = parameters[0];
    const double* P_next = parameters[1];
    double dt = parameters[2][0];

    if (dt < 1e-5) {
      dt = 1e-5;
    }

    const double inv_dt = 1.0 / dt;
    const double inv_dt_sq = inv_dt * inv_dt;

    const double vx_next = (P_next[0] - P_start[0]) * inv_dt;
    const double vy_next = (P_next[1] - P_start[1]) * inv_dt;

    const double ax = (vx_next - vx_start_) * inv_dt;
    const double ay = (vy_next - vy_start_) * inv_dt;

    const double a_norm = std::sqrt(ax * ax + ay * ay + 1e-12);
    const double diff = a_norm - max_a_;

    if (diff <= 0.0) {
      residuals[0] = 0.0;
      if (jacobians != nullptr) {
        zero_block(jacobians[0], 2);
        zero_block(jacobians[1], 2);
        zero_block(jacobians[2], 1);
      }
      return true;
    }

    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      const double w_norm = weight_ / a_norm;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = 0.0;
        jacobians[0][1] = 0.0;
      }

      if (jacobians[1] != nullptr) {
        jacobians[1][0] = w_norm * (ax * inv_dt_sq);
        jacobians[1][1] = w_norm * (ay * inv_dt_sq);
      }

      if (jacobians[2] != nullptr) {
        const double d_ax_dt = (-2.0 * vx_next + vx_start_) * inv_dt_sq;
        const double d_ay_dt = (-2.0 * vy_next + vy_start_) * inv_dt_sq;
        jacobians[2][0] = w_norm * (ax * d_ax_dt + ay * d_ay_dt);
      }
    }

    return true;
  }

private:
  double weight_;
  double max_a_;
  double vx_start_;
  double vy_start_;
};

class StartAngularAccelerationCost : public ceres::SizedCostFunction<1, 1, 1, 1> {
public:
  StartAngularAccelerationCost(double weight, double max_alpha, double current_omega)
      : weight_(weight), max_alpha_(max_alpha), omega_start_(current_omega) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double th_start = parameters[0][0];
    const double th_next = parameters[1][0];
    double dt = parameters[2][0];

    if (dt < 1e-5) {
      dt = 1e-5;
    }

    const double delta_th = normalize_angle_double(th_next - th_start);
    const double inv_dt = 1.0 / dt;
    const double inv_dt2 = inv_dt * inv_dt;

    const double omega_next = delta_th * inv_dt;
    const double alpha = (omega_next - omega_start_) * inv_dt;

    const double abs_alpha = std::abs(alpha);
    const double diff = abs_alpha - max_alpha_;

    if (diff <= 0.0) {
      residuals[0] = 0.0;
      if (jacobians != nullptr) {
        zero_block(jacobians[0], 1);
        zero_block(jacobians[1], 1);
        zero_block(jacobians[2], 1);
      }
      return true;
    }

    const double sign = alpha > 0.0 ? 1.0 : -1.0;
    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      const double w = weight_ * sign;
      const double d_alpha_dth = inv_dt2;
      const double d_alpha_dt =
          -2.0 * delta_th * inv_dt * inv_dt2 + omega_start_ * inv_dt2;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = 0.0;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = w * d_alpha_dth;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = w * d_alpha_dt;
      }
    }

    return true;
  }

private:
  double weight_;
  double max_alpha_;
  double omega_start_;
};

class SegmentAccelerationCost : public ceres::SizedCostFunction<1, 2, 2, 2, 1, 1> {
public:
  SegmentAccelerationCost(double weight, double max_a)
      : weight_(weight), max_a_(max_a) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double* A = parameters[0];
    const double* B = parameters[1];
    const double* C = parameters[2];
    double dt1 = parameters[3][0];
    double dt2 = parameters[4][0];

    if (dt1 < 1e-5 || dt2 < 1e-5) {
      dt1 = std::fmax(dt1, 1e-5);
      dt2 = std::fmax(dt2, 1e-5);
    }

    const double vx1 = (B[0] - A[0]) / dt1;
    const double vy1 = (B[1] - A[1]) / dt1;

    const double vx2 = (C[0] - B[0]) / dt2;
    const double vy2 = (C[1] - B[1]) / dt2;

    const double dt_avg = (dt1 + dt2) / 2.0;
    const double inv_dt_avg = 1.0 / dt_avg;

    const double ax = (vx2 - vx1) * inv_dt_avg;
    const double ay = (vy2 - vy1) * inv_dt_avg;

    const double a_sq = ax * ax + ay * ay;
    const double a_norm = std::sqrt(a_sq + 1e-12);
    const double diff = a_norm - max_a_;

    if (diff <= 0.0) {
      residuals[0] = 0.0;
      if (jacobians != nullptr) {
        zero_block(jacobians[0], 2);
        zero_block(jacobians[1], 2);
        zero_block(jacobians[2], 2);
        zero_block(jacobians[3], 1);
        zero_block(jacobians[4], 1);
      }
      return true;
    }

    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      const double w_norm = weight_ / a_norm;

      const double dax_dA = inv_dt_avg / dt1;
      const double dax_dC = inv_dt_avg / dt2;
      const double dax_dB = inv_dt_avg * (-1.0 / dt2 - 1.0 / dt1);

      const double d_ax_dt1 = -ax / (2.0 * dt_avg) + inv_dt_avg * (vx1 / dt1);
      const double d_ay_dt1 = -ay / (2.0 * dt_avg) + inv_dt_avg * (vy1 / dt1);

      const double d_ax_dt2 = -ax / (2.0 * dt_avg) - inv_dt_avg * (vx2 / dt2);
      const double d_ay_dt2 = -ay / (2.0 * dt_avg) - inv_dt_avg * (vy2 / dt2);

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = w_norm * (ax * dax_dA);
        jacobians[0][1] = w_norm * (ay * dax_dA);
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = w_norm * (ax * dax_dB);
        jacobians[1][1] = w_norm * (ay * dax_dB);
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = w_norm * (ax * dax_dC);
        jacobians[2][1] = w_norm * (ay * dax_dC);
      }
      if (jacobians[3] != nullptr) {
        jacobians[3][0] = w_norm * (ax * d_ax_dt1 + ay * d_ay_dt1);
      }
      if (jacobians[4] != nullptr) {
        jacobians[4][0] = w_norm * (ax * d_ax_dt2 + ay * d_ay_dt2);
      }
    }

    return true;
  }

private:
  double weight_;
  double max_a_;
};

class SegmentAngularAccelerationCost
    : public ceres::SizedCostFunction<1, 1, 1, 1, 1, 1> {
public:
  SegmentAngularAccelerationCost(double weight, double max_alpha)
      : weight_(weight), max_alpha_(max_alpha) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double th_start = parameters[0][0];
    const double th_mid = parameters[1][0];
    const double th_end = parameters[2][0];
    double dt1 = parameters[3][0];
    double dt2 = parameters[4][0];

    if (std::abs(dt1) < 1e-9 || std::abs(dt2) < 1e-9) {
      return false;
    }

    const double dt = dt1 + dt2;
    const double inv_dt = 1.0 / dt;
    const double inv_dt1 = 1.0 / dt1;
    const double inv_dt2 = 1.0 / dt2;

    const double delta_theta1 = normalize_angle_double(th_mid - th_start);
    const double delta_theta2 = normalize_angle_double(th_end - th_mid);

    const double omega1 = delta_theta1 * inv_dt1;
    const double omega2 = delta_theta2 * inv_dt2;

    const double alpha = 2.0 * (omega2 - omega1) * inv_dt;

    double diff = 0.0;
    if (alpha > max_alpha_) {
      diff = alpha - max_alpha_;
    } else if (alpha < -max_alpha_) {
      diff = alpha + max_alpha_;
    }

    residuals[0] = weight_ * diff;

    if (jacobians != nullptr) {
      if (diff == 0.0) {
        zero_block(jacobians[0], 1);
        zero_block(jacobians[1], 1);
        zero_block(jacobians[2], 1);
        zero_block(jacobians[3], 1);
        zero_block(jacobians[4], 1);
        return true;
      }

      const double factor = 2.0 * weight_ * inv_dt;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = factor * inv_dt1;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = -factor * (inv_dt1 + inv_dt2);
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = factor * inv_dt2;
      }
      if (jacobians[3] != nullptr) {
        jacobians[3][0] = factor * omega1 * inv_dt1 - weight_ * alpha * inv_dt;
      }
      if (jacobians[4] != nullptr) {
        jacobians[4][0] = -factor * omega2 * inv_dt2 - weight_ * alpha * inv_dt;
      }
    }

    return true;
  }

private:
  double weight_;
  double max_alpha_;
};

class SegmentKinematicsCost : public ceres::SizedCostFunction<1, 2, 1, 2, 1> {
public:
  explicit SegmentKinematicsCost(double weight) : weight_(weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double x1 = parameters[0][0];
    const double y1 = parameters[0][1];
    const double angle1 = parameters[1][0];
    const double x2 = parameters[2][0];
    const double y2 = parameters[2][1];
    const double angle2 = parameters[3][0];

    const double c1 = std::cos(angle1);
    const double s1 = std::sin(angle1);
    const double c2 = std::cos(angle2);
    const double s2 = std::sin(angle2);

    const double dx = x2 - x1;
    const double dy = y2 - y1;

    const double cos_sum = c1 + c2;
    const double sin_sum = s1 + s2;

    const double error = cos_sum * dy - sin_sum * dx;
    residuals[0] = weight_ * error;

    if (jacobians != nullptr) {
      const double w_sin_sum = weight_ * sin_sum;
      const double w_cos_sum = weight_ * cos_sum;

      if (jacobians[0] != nullptr) {
        jacobians[0][0] = w_sin_sum;
        jacobians[0][1] = -w_cos_sum;
      }
      if (jacobians[1] != nullptr) {
        const double term = -(dy * s1 + dx * c1);
        jacobians[1][0] = weight_ * term;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = -w_sin_sum;
        jacobians[2][1] = w_cos_sum;
      }
      if (jacobians[3] != nullptr) {
        const double term = -(dy * s2 + dx * c2);
        jacobians[3][0] = weight_ * term;
      }
    }

    return true;
  }

private:
  double weight_;
};

class SegmentHeadingCost : public ceres::SizedCostFunction<1, 2, 1, 2> {
public:
  explicit SegmentHeadingCost(double weight) : weight_(weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double x1 = parameters[0][0];
    const double y1 = parameters[0][1];
    const double theta1 = parameters[1][0];
    const double x2 = parameters[2][0];
    const double y2 = parameters[2][1];

    const double dx = x2 - x1;
    const double dy = y2 - y1;

    const double c1 = std::cos(theta1);
    const double s1 = std::sin(theta1);

    const double dot = dx * c1 + dy * s1;

    if (dot >= 0.0) {
      residuals[0] = 0.0;
      if (jacobians != nullptr) {
        zero_block(jacobians[0], 2);
        zero_block(jacobians[1], 1);
        zero_block(jacobians[2], 2);
      }
      return true;
    }

    residuals[0] = weight_ * (-dot);

    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        jacobians[0][0] = weight_ * c1;
        jacobians[0][1] = weight_ * s1;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = weight_ * (dx * s1 - dy * c1);
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = -weight_ * c1;
        jacobians[2][1] = -weight_ * s1;
      }
    }

    return true;
  }

private:
  double weight_;
};

class SegmentAngularSmoothingCost : public ceres::SizedCostFunction<1, 1, 1, 1> {
public:
  explicit SegmentAngularSmoothingCost(double weight) : weight_(weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double theta1 = parameters[0][0];
    const double theta2 = parameters[1][0];
    const double dt = parameters[2][0];

    if (std::abs(dt) < 1e-9) {
      return false;
    }

    const double delta_theta = normalize_angle_double(theta2 - theta1);
    const double inv_dt = 1.0 / dt;

    residuals[0] = weight_ * delta_theta * inv_dt;

    if (jacobians != nullptr) {
      if (jacobians[0] != nullptr) {
        jacobians[0][0] = -weight_ * inv_dt;
      }
      if (jacobians[1] != nullptr) {
        jacobians[1][0] = weight_ * inv_dt;
      }
      if (jacobians[2] != nullptr) {
        jacobians[2][0] = -residuals[0] * inv_dt;
      }
    }

    return true;
  }

private:
  double weight_;
};

class SegmentTimeCost : public ceres::SizedCostFunction<1, 1> {
public:
  explicit SegmentTimeCost(double weight) : weight_(weight) {}

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double dt = parameters[0][0];
    residuals[0] = weight_ * dt;

    if (jacobians != nullptr && jacobians[0] != nullptr) {
      jacobians[0][0] = weight_;
    }
    return true;
  }

private:
  double weight_;
};

class SegmentCircleObstaclesCost : public ceres::CostFunction {
public:
  SegmentCircleObstaclesCost(std::vector<double> obstacles_x,
                             std::vector<double> obstacles_y,
                             std::vector<double> obstacles_r, double weight,
                             double safety_radius)
      : obstacles_x_(std::move(obstacles_x)),
        obstacles_y_(std::move(obstacles_y)),
        obstacles_r_(std::move(obstacles_r)),
        weight_(weight),
        safety_radius_(safety_radius) {
    set_num_residuals(static_cast<int>(obstacles_r_.size()));
    mutable_parameter_block_sizes()->push_back(2);
    mutable_parameter_block_sizes()->push_back(2);
  }

  bool Evaluate(double const* const* parameters, double* residuals,
                double** jacobians) const override {
    const double* A = parameters[0];
    const double* B = parameters[1];

    const double AB_x = B[0] - A[0];
    const double AB_y = B[1] - A[1];
    const double AB_len_sq = std::fmax(AB_x * AB_x + AB_y * AB_y, 1e-10);

    for (std::size_t i = 0; i < obstacles_r_.size(); ++i) {
      const double AO_x = obstacles_x_[i] - A[0];
      const double AO_y = obstacles_y_[i] - A[1];

      double t = (AO_x * AB_x + AO_y * AB_y) / AB_len_sq;
      t = std::fmin(1.0, std::fmax(0.0, t));

      const double O1O_x = AO_x - t * AB_x;
      const double O1O_y = AO_y - t * AB_y;

      const double O1O_len =
          std::sqrt(O1O_x * O1O_x + O1O_y * O1O_y + 1e-10);

      const double errors = obstacles_r_[i] + safety_radius_ - O1O_len;
      const bool active = errors > 0.0;

      residuals[i] = active ? weight_ * errors : 0.0;

      if (jacobians != nullptr) {
        const double j_scaler = active ? weight_ : 0.0;
        const double inv_dist = 1.0 / O1O_len;
        const double d_hat_x = O1O_x * inv_dist;
        const double d_hat_y = O1O_y * inv_dist;

        if (jacobians[0] != nullptr) {
          jacobians[0][2 * i + 0] = j_scaler * (1.0 - t) * d_hat_x;
          jacobians[0][2 * i + 1] = j_scaler * (1.0 - t) * d_hat_y;
        }
        if (jacobians[1] != nullptr) {
          jacobians[1][2 * i + 0] = j_scaler * t * d_hat_x;
          jacobians[1][2 * i + 1] = j_scaler * t * d_hat_y;
        }
      }
    }

    return true;
  }

private:
  std::vector<double> obstacles_x_;
  std::vector<double> obstacles_y_;
  std::vector<double> obstacles_r_;
  double weight_;
  double safety_radius_;
};

struct SegmentLineObstaclesCostFunctor {
  SegmentLineObstaclesCostFunctor(std::vector<double> C_x,
                                  std::vector<double> C_y,
                                  std::vector<double> D_x,
                                  std::vector<double> D_y, double weight,
                                  double safety_radius, double softmin_alpha)
      : C_x(std::move(C_x)), C_y(std::move(C_y)),
        D_x(std::move(D_x)), D_y(std::move(D_y)),
        weight(weight), safety_radius(safety_radius),
        softmin_alpha(softmin_alpha) {}

  template <typename T>
  bool operator()(T const* const* parameters, T* residuals) const {
    const T* A = parameters[0];
    const T* B = parameters[1];

    for (std::size_t i = 0; i < C_x.size(); ++i) {
      T d1, u1_x, u1_y, t1;
      T d2, u2_x, u2_y, t2;
      T d3, u3_x, u3_y, t3;
      T d4, u4_x, u4_y, t4;

      point_segment_distance(A[0], A[1], T(C_x[i]), T(C_y[i]), T(D_x[i]),
                             T(D_y[i]), &d1, &u1_x, &u1_y, &t1);
      point_segment_distance(B[0], B[1], T(C_x[i]), T(C_y[i]), T(D_x[i]),
                             T(D_y[i]), &d2, &u2_x, &u2_y, &t2);
      point_segment_distance(T(C_x[i]), T(C_y[i]), A[0], A[1], B[0], B[1], &d3,
                             &u3_x, &u3_y, &t3);
      point_segment_distance(T(D_x[i]), T(D_y[i]), A[0], A[1], B[0], B[1], &d4,
                             &u4_x, &u4_y, &t4);

      const T scaled1 = T(softmin_alpha) * d1;
      const T scaled2 = T(softmin_alpha) * d2;
      const T scaled3 = T(softmin_alpha) * d3;
      const T scaled4 = T(softmin_alpha) * d4;

      const T max12 = ceres::fmax(scaled1, scaled2);
      const T max34 = ceres::fmax(scaled3, scaled4);
      const T max_scaled_d = ceres::fmax(max12, max34);

      const T exp1 = ceres::exp(scaled1 - max_scaled_d);
      const T exp2 = ceres::exp(scaled2 - max_scaled_d);
      const T exp3 = ceres::exp(scaled3 - max_scaled_d);
      const T exp4 = ceres::exp(scaled4 - max_scaled_d);

      const T sum_exp = exp1 + exp2 + exp3 + exp4;
      const T d_min =
          (ceres::log(sum_exp) + max_scaled_d) / T(softmin_alpha);

      const T errors = T(safety_radius) - d_min;
      residuals[i] = errors > T(0.0) ? T(weight) * errors : T(0.0);
    }

    return true;
  }

  std::vector<double> C_x, C_y, D_x, D_y;
  double weight, safety_radius, softmin_alpha;
};

class PySegmentVelocityCost {
public:
  PySegmentVelocityCost(double weight, double max_v) {
    cost_func_ =
        std::unique_ptr<ceres::CostFunction>(new SegmentVelocityCost(weight, max_v));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentAngularVelocityCost {
public:
  PySegmentAngularVelocityCost(double weight, double max_omega) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(
        new SegmentAngularVelocityCost(weight, max_omega));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PyStartAccelerationCost {
public:
  PyStartAccelerationCost(double weight, double max_a,
                          std::pair<double, double> current_v) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(
        new StartAccelerationCost(weight, max_a, current_v));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PyStartAngularAccelerationCost {
public:
  PyStartAngularAccelerationCost(double weight, double max_alpha,
                                 double current_omega) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(
        new StartAngularAccelerationCost(weight, max_alpha, current_omega));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentAccelerationCost {
public:
  PySegmentAccelerationCost(double weight, double max_a) {
    cost_func_ =
        std::unique_ptr<ceres::CostFunction>(new SegmentAccelerationCost(weight, max_a));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentAngularAccelerationCost {
public:
  PySegmentAngularAccelerationCost(double weight, double max_alpha) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(
        new SegmentAngularAccelerationCost(weight, max_alpha));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentKinematicsCost {
public:
  explicit PySegmentKinematicsCost(double weight) {
    cost_func_ =
        std::unique_ptr<ceres::CostFunction>(new SegmentKinematicsCost(weight));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentHeadingCost {
public:
  explicit PySegmentHeadingCost(double weight) {
    cost_func_ =
        std::unique_ptr<ceres::CostFunction>(new SegmentHeadingCost(weight));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentAngularSmoothingCost {
public:
  explicit PySegmentAngularSmoothingCost(double weight) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(
        new SegmentAngularSmoothingCost(weight));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentTimeCost {
public:
  explicit PySegmentTimeCost(double weight) {
    cost_func_ =
        std::unique_ptr<ceres::CostFunction>(new SegmentTimeCost(weight));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentCircleObstaclesCost {
public:
  PySegmentCircleObstaclesCost(std::vector<double> obstacles_x,
                               std::vector<double> obstacles_y,
                               std::vector<double> obstacles_r, double weight,
                               double safety_radius) {
    cost_func_ = std::unique_ptr<ceres::CostFunction>(new SegmentCircleObstaclesCost(
        std::move(obstacles_x), std::move(obstacles_y), std::move(obstacles_r),
        weight, safety_radius));
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

class PySegmentLineObstaclesCost {
public:
  PySegmentLineObstaclesCost(std::vector<double> C_x, std::vector<double> C_y,
                             std::vector<double> D_x, std::vector<double> D_y,
                             double weight, double safety_radius,
                             double softmin_alpha) {
    using DynCost = ceres::DynamicAutoDiffCostFunction<SegmentLineObstaclesCostFunctor, 4>;

    auto* functor = new SegmentLineObstaclesCostFunctor(
        std::move(C_x), std::move(C_y), std::move(D_x), std::move(D_y),
        weight, safety_radius, softmin_alpha);

    auto* cost = new DynCost(functor);
    cost->SetNumResiduals(static_cast<int>(functor->C_x.size()));
    cost->AddParameterBlock(2);
    cost->AddParameterBlock(2);

    cost_func_ = std::unique_ptr<ceres::CostFunction>(cost);
  }

  bool evaluate(std::size_t n_params, std::uintptr_t parameters_ptr,
                std::uintptr_t residuals_ptr, std::uintptr_t jacobians_ptr) {
    evaluate_cost(cost_func_.get(), n_params, parameters_ptr, residuals_ptr,
                  jacobians_ptr);
    return true;
  }

private:
  std::unique_ptr<ceres::CostFunction> cost_func_;
};

PYBIND11_MODULE(goof_costs, m) {
  py::class_<PySegmentVelocityCost>(m, "SegmentVelocityCostExt")
      .def(py::init<double, double>())
      .def("evaluate", &PySegmentVelocityCost::evaluate);

  py::class_<PySegmentAngularVelocityCost>(m, "SegmentAngularVelocityCostExt")
      .def(py::init<double, double>())
      .def("evaluate", &PySegmentAngularVelocityCost::evaluate);

  py::class_<PyStartAccelerationCost>(m, "StartAccelerationCostExt")
      .def(py::init<double, double, std::pair<double, double>>())
      .def("evaluate", &PyStartAccelerationCost::evaluate);

  py::class_<PyStartAngularAccelerationCost>(m, "StartAngularAccelerationCostExt")
      .def(py::init<double, double, double>())
      .def("evaluate", &PyStartAngularAccelerationCost::evaluate);

  py::class_<PySegmentAccelerationCost>(m, "SegmentAccelerationCostExt")
      .def(py::init<double, double>())
      .def("evaluate", &PySegmentAccelerationCost::evaluate);

  py::class_<PySegmentAngularAccelerationCost>(
      m, "SegmentAngularAccelerationCostExt")
      .def(py::init<double, double>())
      .def("evaluate", &PySegmentAngularAccelerationCost::evaluate);

  py::class_<PySegmentKinematicsCost>(m, "SegmentKinematicsCostExt")
      .def(py::init<double>())
      .def("evaluate", &PySegmentKinematicsCost::evaluate);

  py::class_<PySegmentHeadingCost>(m, "SegmentHeadingCostExt")
      .def(py::init<double>())
      .def("evaluate", &PySegmentHeadingCost::evaluate);

  py::class_<PySegmentAngularSmoothingCost>(m, "SegmentAngularSmoothingCostExt")
      .def(py::init<double>())
      .def("evaluate", &PySegmentAngularSmoothingCost::evaluate);

  py::class_<PySegmentTimeCost>(m, "SegmentTimeCostExt")
      .def(py::init<double>())
      .def("evaluate", &PySegmentTimeCost::evaluate);

  py::class_<PySegmentCircleObstaclesCost>(m, "SegmentCircleObstaclesCostExt")
      .def(py::init<std::vector<double>, std::vector<double>,
                    std::vector<double>, double, double>())
      .def("evaluate", &PySegmentCircleObstaclesCost::evaluate);

  py::class_<PySegmentLineObstaclesCost>(m, "SegmentLineObstaclesCostExt")
      .def(py::init<std::vector<double>, std::vector<double>,
                    std::vector<double>, std::vector<double>, double, double,
                    double>())
      .def("evaluate", &PySegmentLineObstaclesCost::evaluate);
}
