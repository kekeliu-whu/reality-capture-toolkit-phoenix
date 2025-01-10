#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

struct GpsFactor {
  GpsFactor(const Sophus::SE3d &pose_lidar2imu, const Eigen::Vector3d &t_gps,
            double t, double scale)
      : pose_lidar2imu_(pose_lidar2imu), t_gps_(t_gps), t_(t), scale_(scale) {}

  template <typename T>
  bool operator()(const T *const p_a_ptr, const T *const q_a_ptr,
                  const T *const p_b_ptr, const T *const q_b_ptr,
                  T *residual) const {
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_a(p_a_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_a(q_a_ptr);
    Eigen::Map<const Eigen::Matrix<T, 3, 1>> p_b(p_b_ptr);
    Eigen::Map<const Eigen::Quaternion<T>> q_b(q_b_ptr);

    Sophus::SE3<T> pose_a = Sophus::SE3<T>(q_a, p_a);
    Sophus::SE3<T> pose_b = Sophus::SE3<T>(q_b, p_b);

    Sophus::SE3<T> pose_interp =
        pose_a * Sophus::SE3<T>::exp(t_ * (pose_a.inverse() * pose_b).log());

    Sophus::SE3<T> pose_imu = pose_lidar2imu_.inverse().cast<T>() * pose_interp;

    Eigen::Map<Eigen::Matrix<T, 3, 1>>{residual} =
        (pose_imu.translation() - t_gps_.cast<T>()) / scale_;
    return true;
  }

  static ceres::CostFunction *Create(const Sophus::SE3d &pose_lidar2imu,
                                     const Eigen::Vector3d &t_gps, double t,
                                     double scale) {
    return new ceres::AutoDiffCostFunction<GpsFactor, 3, 3, 4, 3, 4>(
        new GpsFactor(pose_lidar2imu, t_gps, t, scale));
  }

private:
  Eigen::Vector3d t_gps_;
  Sophus::SE3d pose_lidar2imu_;
  double t_;
  double scale_;
};
