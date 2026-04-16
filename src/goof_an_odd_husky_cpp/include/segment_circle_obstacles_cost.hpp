#pragma once
#ifndef INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#define INCLUDE_SEGMENT_VELOCITY_COST_HPP_
#include <ceres/ceres.h>

struct SegmentCircleObstaclesCost {
  SegmentCircleObstaclesCost(std::vector<double> ox,
                             std::vector<double> oy,
                             std::vector<double> r,
                             double w,
                             double safety)
      : ox(std::move(ox)), oy(std::move(oy)),
        r(std::move(r)), weight(w), safety(safety) {}

  template <typename T>
  bool operator()(const T* A, const T* B, T* residuals) const {

    T ABx = B[0] - A[0];
    T ABy = B[1] - A[1];
    T len_sq = ceres::max(ABx*ABx + ABy*ABy, T(1e-10));

    for (size_t i = 0; i < ox.size(); ++i) {
      T AOx = T(ox[i]) - A[0];
      T AOy = T(oy[i]) - A[1];

      T t = (AOx*ABx + AOy*ABy) / len_sq;
      t = ceres::max(T(0), ceres::min(T(1), t));

      T dx = AOx - t*ABx;
      T dy = AOy - t*ABy;

      T dist = ceres::sqrt(dx*dx + dy*dy + T(1e-10));

      T err = T(r[i] + safety) - dist;
      residuals[i] = err > T(0) ? T(weight) * err : T(0);
    }
    return true;
  }

  std::vector<double> ox, oy, r;
  double weight, safety;
};

#endif // INCLUDE_SEGMENT_VELOCITY_COST_HPP_
