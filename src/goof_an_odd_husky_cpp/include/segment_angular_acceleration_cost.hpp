#pragma once
#ifndef INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#define INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#include <ceres/ceres.h>

struct SegmentAngularAccelerationCost {
  SegmentAngularAccelerationCost(double w, double max_a)
      : weight(w), max_alpha(max_a) {}

  template <typename T>
  bool operator()(const T* A, const T* B, const T* C,
                  const T* dt1, const T* dt2,
                  T* r) const {

    if (ceres::abs(dt1[0]) < T(1e-9) ||
        ceres::abs(dt2[0]) < T(1e-9))
      return false;

    T inv_dt1 = T(1) / dt1[0];
    T inv_dt2 = T(1) / dt2[0];
    T inv_dt = T(1) / (dt1[0] + dt2[0]);

    T d1 = normalize_angle(B[0] - A[0]);
    T d2 = normalize_angle(C[0] - B[0]);

    T w1 = d1 * inv_dt1;
    T w2 = d2 * inv_dt2;

    T alpha = T(2) * (w2 - w1) * inv_dt;

    T diff = T(0);
    if (alpha > T(max_alpha)) diff = alpha - T(max_alpha);
    else if (alpha < -T(max_alpha)) diff = alpha + T(max_alpha);

    r[0] = T(weight) * diff;
    return true;
  }

  double weight, max_alpha;
};

#endif // INCLUDE_SEGMENT_VELOCITY_COST_HPP_
