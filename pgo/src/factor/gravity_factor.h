#pragma once

#include <ceres/ceres.h>
#include <Eigen/Dense>

// Gravity alignment constraint as a quaternion-only rotation residual.
//
// Given a pose's measured gravity direction in the local lidar/body frame and
// the reference world gravity direction, this functor rotates the measured
// local direction into world frame and penalizes deviation from the reference.
//
//   residual = weight * ( R(q) * g_body - g_world )
//
// This actively drives roll/pitch toward zero alignment with the physical
// gravity direction, leaving yaw unconstrained.
struct GravityCostFunctor {
  GravityCostFunctor(const Eigen::Vector3d& body_gravity,
                     const Eigen::Vector3d& world_gravity,
                     double error_deg) {
    body_gravity_  = body_gravity.normalized();
    world_gravity_ = world_gravity.normalized();
    double sigma_rad = error_deg * M_PI / 180.0;
    weight_ = 1.0 / sigma_rad;
  }

  // T_now_ptr: Sophus SE3 layout  [qx, qy, qz, qw, tx, ty, tz]  (7 elements)
  template <typename T>
  bool operator()(const T* const T_now_ptr, T* residual) const {
    // Quaternion from Sophus order [qx,qy,qz,qw] to Eigen order [w,x,y,z]
    Eigen::Quaternion<T> q(T_now_ptr[3], T_now_ptr[0], T_now_ptr[1], T_now_ptr[2]);

    Eigen::Matrix<T, 3, 1> body_g(T(body_gravity_.x()),
                                  T(body_gravity_.y()),
                                  T(body_gravity_.z()));
    Eigen::Matrix<T, 3, 1> estimated_world_g = q * body_g;

    residual[0] = T(weight_) * (estimated_world_g[0] - T(world_gravity_.x()));
    residual[1] = T(weight_) * (estimated_world_g[1] - T(world_gravity_.y()));
    residual[2] = T(weight_) * (estimated_world_g[2] - T(world_gravity_.z()));
    return true;
  }

  static ceres::CostFunction* Create(const Eigen::Vector3d& body_gravity,
                                     const Eigen::Vector3d& world_gravity,
                                     double error_deg) {
    return new ceres::AutoDiffCostFunction<GravityCostFunctor, 3, 7>(
        new GravityCostFunctor(body_gravity, world_gravity, error_deg));
  }

 private:
  Eigen::Vector3d body_gravity_;
  Eigen::Vector3d world_gravity_;
  double weight_;
};
