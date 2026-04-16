#pragma once
#ifndef INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#define INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#include <ceres/ceres.h>

struct SegmentAngularVelocityCost {
  SegmentAngularVelocityCost(double w, double max_w)
      : weight(w), max_omega(max_w) {}

  template <typename T>
  bool operator()(const T* A, const T* B, const T* dt, T* r) const {
    if (ceres::abs(dt[0]) < T(1e-9)) return false;

    T delta = normalize_angle(B[0] - A[0]);
    T omega = delta / dt[0];

    T diff = T(0);
    if (omega > T(max_omega)) diff = omega - T(max_omega);
    else if (omega < -T(max_omega)) diff = omega + T(max_omega);

    r[0] = T(weight) * diff;
    return true;
  }

  double weight, max_omega;
};

#endif // INCLUDE_SEGMENT_VELOCITY_COST_HPP_
