#pragma once
#ifndef INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#define INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#include <ceres/ceres.h>

struct SegmentAccelerationCost {
  SegmentAccelerationCost(double w, double max_a)
      : weight(w), max_a(max_a) {}

  template <typename T>
  bool operator()(const T* A, const T* B, const T* C,
                  const T* dt1, const T* dt2,
                  T* r) const {

    T t1 = ceres::max(dt1[0], T(1e-5));
    T t2 = ceres::max(dt2[0], T(1e-5));

    T vx1 = (B[0] - A[0]) / t1;
    T vy1 = (B[1] - A[1]) / t1;

    T vx2 = (C[0] - B[0]) / t2;
    T vy2 = (C[1] - B[1]) / t2;

    T dt_avg = (t1 + t2) * T(0.5);

    T ax = (vx2 - vx1) / dt_avg;
    T ay = (vy2 - vy1) / dt_avg;

    T norm = ceres::sqrt(ax * ax + ay * ay + T(1e-12));

    T diff = norm - T(max_a);
    r[0] = diff > T(0) ? T(weight) * diff : T(0);

    return true;
  }

  double weight, max_a;
};

#endif // INCLUDE_SEGMENT_VELOCITY_COST_HPP_
