//
// Created by youyuan on 24-2-4.
//
#pragma once

#define EIGEN_STACK_ALLOCATION_LIMIT 10048576
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <chrono>
#include <deque>

namespace lixel
{

// TODO: Determine the double and float in calculation
using FloatDataType = double;
using IntDataType = int32_t;

using Vec3 = Eigen::Matrix<FloatDataType, 3, 1>;
using V3D = Eigen::Vector3d;
using V3F = Eigen::Vector3f;

using Mat3 = Eigen::Matrix<FloatDataType, 3, 3>;
using M3D = Eigen::Matrix3d;
using M3F = Eigen::Matrix3f;

using Mat4 = Eigen::Matrix<FloatDataType, 4, 4>;
using M4D = Eigen::Matrix4d;
using M4F = Eigen::Matrix4f;

using QUAT = Eigen::Quaternion<FloatDataType>;
using QUATD = Eigen::Quaterniond;
using QUATF = Eigen::Quaternionf;

struct EIGEN_ALIGN16 PointIRT : public pcl::_PointXYZ
{
  float intensity;
  double timestamp;
  uint16_t ring;

  inline PointIRT(const _PointXYZ &p)
  {
    x = p.x;
    y = p.y;
    z = p.z;
    data[3] = 1.0f;
  }

  inline PointIRT()
  {
    x = y = z = 0.0f;
    data[3] = 1.0f;
  }

  inline PointIRT(float _x, float _y, float _z)
  {
    x = _x;
    y = _y;
    z = _z;
    data[3] = 1.0f;
  }

  friend std::ostream &operator<<(std::ostream &os, const PointIRT &p);
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};

typedef PointIRT PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointXYZINormal PointXYZINormal;
typedef pcl::PointCloud<PointXYZINormal> PointCloudXYZINormal;

enum MEAS_TYPE
{
  LIDAR,
  GNSS,
  VISUAL,
  STATIC
};

typedef struct PosAtt
{
  double timestamp;
  V3F pos;
  V3F vel;
  QUATF quat;
} PosAtt;

typedef std::deque<PosAtt> StatePredict;

typedef struct AttributeImu
{
  uint32_t sweep_id;
  double timestamp;
  float dt;
  V3F gyro_true;
  V3F acc_true;
  V3F gyro_world;
  V3F acc_world;
} AttributeImu;

typedef std::deque<AttributeImu> AttributePredict;

typedef struct AttributeIterate
{
  uint32_t iter_num;
  V3F pos_tol;
  V3F rot_tol;
  float std_dev;
  V3D pos_update;
  V3D rot_update;
  V3D vel_update;
  V3D gyro_bias_update;
  V3D acc_bias_update;

  V3D rot_std;
  V3D pos_std;
  V3D vel_std;
  V3D gyro_bias_std;
  V3D acc_bias_std;
  V3D gravity_std;
} AttributeIterate;

typedef struct AttributeJacobi
{
  V3F point_eig;
  float flat_ness;
  V3F norm_eig;
  float smooth_ness;
  uint32_t use_point_num;
  uint32_t total_point_num;
  float overlap_radio;
} AttributeJacobi;

typedef struct AttributeIESKF
{
  uint32_t sweep_id;
  double timestamp;
  float downsample_dis;
  StatePredict state_predict;
  AttributePredict attritube_predict;
  // FullStateMsg state_increment;
  AttributeJacobi jacobi;
  AttributeIterate iterate;
} AttributeIESKF;

class TicToc
{
 public:
  TicToc()
  {
    Tic();
  }

  void Tic()
  {
    start_ = std::chrono::system_clock::now();
  }

  double Toc()
  {
    end_ = std::chrono::system_clock::now();
    elapsed_seconds_ = end_ - start_;
    return elapsed_seconds_.count() * 1000;
  }

  double GetLastStop()
  {
    return elapsed_seconds_.count() * 1000;
  }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_, end_;
  std::chrono::duration<double> elapsed_seconds_{};
};

}  // namespace lixel
