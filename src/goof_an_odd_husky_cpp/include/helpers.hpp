#pragma once
#ifndef INCLUDE_HELPERS_HPP_
#define INCLUDE_HELPERS_HPP_

#include <ceres/ceres.h>

template <typename T>
inline T normalize_angle(const T& a) {
  return ceres::atan2(ceres::sin(a), ceres::cos(a));
}

#endif // INCLUDE_HELPERS_HPP_
