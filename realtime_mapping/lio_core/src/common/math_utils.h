//
// Created by youyuan on 24-2-4.
//
#pragma once

#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <chrono>
#include <deque>
#include "common_struct.h"
#include "log/lsLogger.h"

namespace lixel
{

constexpr double PI = 3.14159265358;
constexpr double RAD_2_DEG = 180 / PI;
constexpr double DEG_2_RAD = 1 / RAD_2_DEG;
constexpr float ZERO_TOLERANCE = 1e-7;
constexpr double DEFAULT_GRAVITY = 9.8;
static const Vec3 DEFAULT_GRIVITY_VEC(0, 0, -lixel::DEFAULT_GRAVITY);

inline Mat3 skewSymMatrix(const Vec3 &v)
{
  Mat3 K;
  K << 0.0, -v[2], v[1], v[2], 0.0, -v[0], -v[1], v[0], 0.0;
  return K;
}

inline Mat3 exp(const Vec3 &angle_axis)
{
  FloatDataType angle_axis_norm = angle_axis.norm();
  Mat3 Eye3 = Mat3::Identity();
  if (angle_axis_norm > ZERO_TOLERANCE)
  {
    Vec3 r_axis = angle_axis / angle_axis_norm;
    Mat3 K;
    K << skewSymMatrix(r_axis);
    /// Roderigous Tranformation
    return Eye3 + std::sin(angle_axis_norm) * K + (1.0 - std::cos(angle_axis_norm)) * K * K;
  }
  else
  {
    return Eye3;
  }
}

inline void SVD(const std::vector<V3F> &data, M3F &V, V3F &Sigma)
{
  int numPoints = data.size();
  Eigen::MatrixXf data_matrix(numPoints, 3);
  for (int i = 0; i < numPoints; i++) data_matrix.row(i) = data[i].transpose();
  V3F mean_point = data_matrix.colwise().mean();
  Eigen::JacobiSVD<Eigen::MatrixXf> svd(data_matrix, Eigen::ComputeThinU | Eigen::ComputeFullV);
  V = svd.matrixV();
  Sigma = svd.singularValues().transpose();
}

inline Vec3 log(const Mat3 &R)
{
  FloatDataType theta = (R.trace() > 3.0 - ZERO_TOLERANCE) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
  Vec3 K(R(2, 1) - R(1, 2), R(0, 2) - R(2, 0), R(1, 0) - R(0, 1));
  return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

inline Vec3 rotMtoEuler(const Mat3 &rot)
{
  FloatDataType sy = sqrt(rot(0, 0) * rot(0, 0) + rot(1, 0) * rot(1, 0));
  bool singular = sy < ZERO_TOLERANCE;
  FloatDataType x, y, z;
  if (!singular)
  {
    x = atan2(rot(2, 1), rot(2, 2));
    y = atan2(-rot(2, 0), sy);
    z = atan2(rot(1, 0), rot(0, 0));
  }
  else
  {
    x = atan2(-rot(1, 2), rot(1, 1));
    y = atan2(-rot(2, 0), sy);
    z = 0;
  }
  Vec3 ang(x, y, z);
  return ang;
}
inline double calcuateGravity(const double &latitude)
{
  if (latitude < -lixel::PI || latitude > lixel::PI)
  {
    lslog(LSLOG_INFO) << "Invalid latitude: " << latitude << " is not in the range of [-PI, PI]";
    return lixel::DEFAULT_GRAVITY;
  }
  return 9.780327 * (1 + 5.3024e-3 * pow(sin(latitude), 2) - 5.8e-6 * pow(sin(2 * latitude), 4));
}

}  // namespace lixel