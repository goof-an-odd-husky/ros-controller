#pragma once
#ifndef INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#define INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#include <ceres/ceres.h>

struct SegmentVelocityCost {
  SegmentVelocityCost(double weight, double max_v)
      : weight(weight), max_v(max_v) {}

  template <typename T>
  bool operator()(const T *const A, const T *const B, const T *const dt,
                  T *residual) const {

    if (ceres::abs(dt[0]) < T(1e-9)) {
      return false;
    }

    T AB_x = B[0] - A[0];
    T AB_y = B[1] - A[1];

    T AB_len = ceres::sqrt(AB_x * AB_x + AB_y * AB_y + T(1e-10));
    T v = AB_len / dt[0];

    T diff = T(0.0);
    if (v > T(max_v)) {
      diff = v - T(max_v);
    }

    residual[0] = T(weight) * diff;
    return true;
  }

  double weight;
  double max_v;
};
#endif // INCLUDE_SEGMENT_VELOCITY_COST_HPP_
