#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>
#include <Eigen/Dense>

/**
 * @brief RTK position constraint factor
 * 
 * Constraint a submap's pose based on GNSS/RTK position measurement.
 * Uses WGS84 latitude/longitude/altitude converted to local ENU coordinates.
 */
struct RtkPositionFactor {
  RtkPositionFactor(const Eigen::Vector3d &gps_position_enu,
                    const Eigen::Vector3d &gps_position_std)
      : gps_position_enu_(gps_position_enu),
        gps_position_std_inv_(gps_position_std.cwiseInverse()) {}

  template <typename T>
  bool operator()(const T *const pose_ptr, T *residual) const {
    Eigen::Map<const Sophus::SE3<T>> pose(pose_ptr);
    
    // Extract position from SE3 pose
    Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_vec(residual);
    
    // Calculate residual: (measured_position - estimated_position) * inv_std
    residual_vec = (gps_position_enu_.cast<T>() - pose.translation()).cwiseProduct(
                   gps_position_std_inv_.cast<T>());
    
    return true;
  }

  static ceres::CostFunction *Create(const Eigen::Vector3d &gps_position_enu,
                                     const Eigen::Vector3d &gps_position_std) {
    return new ceres::AutoDiffCostFunction<RtkPositionFactor, 3, 7>(
        new RtkPositionFactor(gps_position_enu, gps_position_std));
  }

 private:
  Eigen::Vector3d gps_position_enu_;     // GNSS position in ENU frame (meters)
  Eigen::Vector3d gps_position_std_inv_;  // 1.0 / position standard deviation
};
