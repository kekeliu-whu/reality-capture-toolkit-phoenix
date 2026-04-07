#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

// Computes the error term for two poses that have a relative pose measurement
// between them. Let the hat variables be the measurement. We have two poses x_a
// and x_b. Through sensor measurements we can measure the transformation of
// frame B w.r.t frame A denoted as t_ab_hat. We can compute an error metric
// between the current estimate of the poses and the measurement.
//
// In this formulation, we have chosen to represent the rigid transformation as
// a Hamiltonian quaternion, q, and position, p. The quaternion ordering is
// [x, y, z, w].
// The estimated measurement is:
//      t_ab = [ p_ab ]  = [ R(q_a)^T * (p_b - p_a) ]
//             [ q_ab ]    [ q_a^{-1] * q_b         ]
//
// where ^{-1} denotes the inverse and R(q) is the rotation matrix for the
// quaternion. Now we can compute an error metric between the estimated and
// measurement transformation. For the orientation error, we will use the
// standard multiplicative error resulting in:
//
//   error = [ p_ab - \hat{p}_ab                 ]
//           [ 2.0 * Vec(q_ab * \hat{q}_ab^{-1}) ]
//
// where Vec(*) returns the vector (imaginary) part of the quaternion. Since
// the measurement has an uncertainty associated with how accurate it is, we
// will weight the errors by the square root of the measurement information
// matrix:
//
//   residuals = I^{1/2) * error
// where I is the information matrix which is the inverse of the covariance.
class PoseGraphEdgeFactor {
public:
  PoseGraphEdgeFactor(Sophus::SE3d T_b2a,
                      Eigen::Matrix<double, 6, 6> sqrt_information)
      : T_b2a_(T_b2a),
        sqrt_information_(std::move(sqrt_information)) {}

  template <typename T>
  bool operator()(const T *const T_a_ptr, const T *const T_b_ptr,
                  T *residuals_ptr) const {
    // Convert input pointers to Sophus::SE3 objects
    Eigen::Map<const Sophus::SE3<T>> T_a(T_a_ptr);
    Eigen::Map<const Sophus::SE3<T>> T_b(T_b_ptr);

    // Compute the relative transformation
    Sophus::SE3<T> T_b2a_estimated = T_a.inverse() * T_b;

    // Compute the error
    Sophus::SE3<T> T_b2a_error =
        T_b2a_estimated * T_b2a_.cast<T>().inverse();

    // Compute the residuals
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals.template head<3>() = T_b2a_error.translation();
    residuals.template tail<3>() =
        T(2.0) * T_b2a_error.so3().unit_quaternion().vec();

    // Apply the square root information matrix
    residuals = sqrt_information_.template cast<T>() * residuals;

    return true;
  }

  static ceres::CostFunction *
  Create(const Sophus::SE3d &T_b2a,
         const Eigen::Matrix<double, 6, 6> &sqrt_information) {
    return new ceres::AutoDiffCostFunction<PoseGraphEdgeFactor, 6, 7, 7>(
        new PoseGraphEdgeFactor(T_b2a, sqrt_information));
  }

private:
  Sophus::SE3d T_b2a_;
  Eigen::Matrix<double, 6, 6> sqrt_information_;
};
