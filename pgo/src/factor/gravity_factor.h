#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

class GravityAlignFactor {
 public:
  GravityAlignFactor(const Sophus::SE3d &T_origin,
                     const Eigen::Vector3d &gravity_vector,
                     Eigen::Matrix<double, 3, 3> sqrt_information)
      : T_origin_(T_origin),
        gravity_vector_(gravity_vector.normalized()),
        sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *const T_now_ptr,
                  T *residuals_ptr) const {
    // Convert input pointers to Sophus::SE3 objects
    Eigen::Map<const Sophus::SE3<T>> T_now(T_now_ptr);

    // Compute the rotation difference between T_origin and T_now
    Sophus::SO3<T> R_origin = T_origin_.so3().cast<T>();
    Sophus::SO3<T> R_now    = T_now.so3();

    // Compute the residual as the difference between the rotated gravity vector and the original gravity vector
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residuals(residuals_ptr);
    // Here "residual * 2" means the angular misalign value
    residuals = T(2.0) * (R_now * R_origin.inverse() * gravity_vector_.cast<T>() - gravity_vector_.cast<T>());

    // Apply the square root information matrix
    residuals = sqrt_information_.template cast<T>() * residuals;

    return true;
  }

  static ceres::CostFunction *
  Create(const Sophus::SE3d &T_origin,
         const Eigen::Vector3d &gravity_vector,
         const Eigen::Matrix<double, 3, 3> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<GravityAlignFactor, 3, 7>(
        new GravityAlignFactor(T_origin, gravity_vector, sqrt_information));
  }

 private:
  Sophus::SE3d T_origin_;
  Eigen::Vector3d gravity_vector_;
  Eigen::Matrix<double, 3, 3> sqrt_information_;
};
