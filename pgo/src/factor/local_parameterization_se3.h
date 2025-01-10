#pragma once

#include <ceres/ceres.h>
#include <sophus/se3.hpp>

class LocalParameterizationSE3 {
 public:
  virtual ~LocalParameterizationSE3() {}

  template <typename T>
  bool Plus(const T *x,
            const T *delta,
            T *x_plus_delta) const {
    const Eigen::Map<const Sophus::SE3<T>> pose{x};
    const Eigen::Map<const Eigen::Matrix<T, 6, 1>> pose_delta_raw{delta};
    Eigen::Map<Sophus::SE3<T>> T_plus_delta(x_plus_delta);
    T_plus_delta = pose * Sophus::SE3<T>::exp(pose_delta_raw);
    return true;
  }

  template <typename T>
  bool Minus(const T *y,
             const T *x,
             T *y_minus_x) const {
    const Eigen::Map<const Sophus::SE3<T>> pose_x{x};
    const Eigen::Map<const Sophus::SE3<T>> pose_y{y};

    Eigen::Map<Eigen::Vector<T, 6>>{y_minus_x} = (pose_x.inverse() * pose_y).log();

    return true;
  }

  static ceres::Manifold *Create() {
    return new ceres::AutoDiffManifold<LocalParameterizationSE3, 7,
                                       6>();
  }
};
