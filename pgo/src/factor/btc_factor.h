#pragma once

#include <ceres/ceres.h>
#include <Sophus/se3.hpp>

/**
 * @brief BTC (Binary Triangle Cluster) constraint factor for loop closure
 *
 * This factor represents a pose-to-pose constraint derived from BTC-based
 * loop closure detection. It constrains the relative pose between two submaps.
 *
 * The residual is the log-map of the pose difference in SE(3).
 */
class BTCFactor {
 public:
  /**
   * @brief Create a cost function for BTC loop closure constraint
   *
   * @param T_source_to_target Relative pose from source to target frame
   * @param information Information matrix (inverse covariance) for the constraint
   * @return Ceres cost function pointer
   */
  static ceres::CostFunction *Create(
      const Sophus::SE3d &T_source_to_target,
      const Eigen::DiagonalMatrix<double, 6> &information) {
    return new ceres::AutoDiffCostFunction<BTCFactor, 6, 7, 7>(
        new BTCFactor(T_source_to_target, information));
  }

 private:
  BTCFactor(const Sophus::SE3d &T_source_to_target,
            const Eigen::DiagonalMatrix<double, 6> &information)
      : T_source_to_target_(T_source_to_target),
        sqrt_information_(information) {}

  template <typename T>
  bool operator()(const T *const pose_target_ptr, const T *const pose_source_ptr,
                  T *residuals_ptr) const {
    // Convert raw pointers to Sophus SE3 poses
    Eigen::Map<const Sophus::SE3<T>> pose_target(pose_target_ptr);
    Eigen::Map<const Sophus::SE3<T>> pose_source(pose_source_ptr);

    // Compute pose difference
    Sophus::SE3<T> T_diff =
        (pose_target.inverse() * pose_source) *
        T_source_to_target_.cast<T>().inverse();

    // Convert to tangent space (log-map) to get 6D residual
    Eigen::Map<Eigen::Matrix<T, 6, 1>> residuals(residuals_ptr);
    residuals = T_diff.log();

    // Apply information weighting
    Eigen::Matrix<double, 6, 6> sqrt_info_matrix =
        sqrt_information_.toDenseMatrix();
    for (int i = 0; i < 6; ++i) {
      residuals(i) *= sqrt_info_matrix(i, i);
    }

    return true;
  }

  Sophus::SE3d T_source_to_target_;
  Eigen::DiagonalMatrix<double, 6> sqrt_information_;
};
