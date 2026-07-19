#pragma once

#include <pcl/common/io.h>
#include <eigen3/Eigen/Dense>

#include "common/common_struct.h"

namespace lixel
{

// Geodetic coordinate system type
enum class GcsType
{
  BEI_JING_54 = 1,
  XI_AN_80,
  WGS_84,
  CGCS_2000
};

enum class SensorDataType
{
  IMU = 0,
  LIDAR = 1,
  MOTOR = 2,
  GNSS = 3,
  IMAGE = 4,
  ODOM = 5
};

struct ImuMsg
{
  using Ptr = std::shared_ptr<ImuMsg>;
  using ConstPtr = std::shared_ptr<const ImuMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuMsg() = default;
  ImuMsg(const double _timestamp, const V3D &_acc, const V3D &_gyro) : timestamp(_timestamp), acc(_acc), gyro(_gyro)
  {
  }

  double timestamp;
  V3D acc;
  V3D gyro;
  SensorDataType sensor_type = SensorDataType::IMU;
};

struct MotorMsg
{
  using Ptr = std::shared_ptr<MotorMsg>;
  using ConstPtr = std::shared_ptr<const MotorMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  MotorMsg() = default;
  MotorMsg(const double _timestamp, const QUATD &_q) : timestamp(_timestamp), q(_q)
  {
  }

  double timestamp = 0.0;
  QUATF q;
  SensorDataType sensor_type = SensorDataType::MOTOR;
};

struct PointCloudMsg
{
  using Ptr = std::shared_ptr<PointCloudMsg>;
  using ConstPtr = std::shared_ptr<const PointCloudMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  PointCloudMsg() = default;
  PointCloudMsg(const double _timestamp, const PointCloud::Ptr &_points) : timestamp(_timestamp), points(_points){};

  double timestamp = 0.0;
  int frame_id;
  PointCloud::Ptr points;
  SensorDataType sensor_type = SensorDataType::LIDAR;
};

struct GnssMsg
{
  using Ptr = std::shared_ptr<GnssMsg>;
  using ConstPtr = std::shared_ptr<const GnssMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  GnssMsg() = default;
  GnssMsg(
      const double _timestamp,
      const int _status,
      const V3D &_lbh,
      const V3D &_pos_cov,
      const double _h_dop,
      const double _v_dop,
      const int _satellites_num,
      const GcsType _gcs_type)
      : timestamp(_timestamp),
        status(_status),
        lbh(_lbh),
        pos_cov(_pos_cov),
        h_dop(_h_dop),
        v_dop(_v_dop),
        satellites_num(_satellites_num),
        gcs_type(_gcs_type){};

  double timestamp;
  int status;
  V3D lbh;
  V3D pos_cov;
  double h_dop;
  double v_dop;
  int satellites_num;
  GcsType gcs_type;
  QUATD q;

  V3D local_xyz;
  V3D local_vel;
  bool valid;
  int reconnect_count;
  SensorDataType sensor_type = SensorDataType::GNSS;
};

struct ImageMsg
{
  using Ptr = std::shared_ptr<ImageMsg>;
  using ConstPtr = std::shared_ptr<const ImageMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImageMsg() = default;
  // ImageMsg(const double _timestamp, const cv::Mat &_image) : timestamp(_timestamp),
  // image(_image){};

  double timestamp;
  // cv::Mat image;
  SensorDataType sensor_type = SensorDataType::IMAGE;
};

struct OdometryMsg
{
  using Ptr = std::shared_ptr<OdometryMsg>;
  using ConstPtr = std::shared_ptr<const OdometryMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  OdometryMsg() = default;
  OdometryMsg(const double _timestamp, const QUATD &_q, const V3D &_p, const V3D &_v)
      : timestamp(_timestamp), q(_q), p(_p), v(_v){};

  double timestamp;
  QUATD q;
  V3D p;
  V3D v;
  SensorDataType sensor_type = SensorDataType::ODOM;
};

struct FullStateMsg
{
  using Ptr = std::shared_ptr<FullStateMsg>;
  using ConstPtr = std::shared_ptr<const FullStateMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  FullStateMsg() = default;
  FullStateMsg(
      const double _timestamp,
      const QUATD &_q,
      const V3D &_p,
      const V3D &_v,
      const V3D &_ba,
      const V3D &_bg,
      const V3D &_gravity)
      : timestamp(_timestamp), q(_q), p(_p), v(_v), ba(_ba), bg(_bg), gravity(_gravity){};

  double timestamp;
  QUATD q;
  V3D p;
  V3D v;
  V3D ba;
  V3D bg;
  V3D gravity;
};

struct DebugMsg
{
  double timestamp;
  double data[30];
};

typedef std::deque<DebugMsg> DebugMsgs;

struct LioResultMsg
{
  using Ptr = std::shared_ptr<LioResultMsg>;
  using ConstPtr = std::shared_ptr<const LioResultMsg>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LioResultMsg() = default;
  LioResultMsg(
      const FullStateMsg &_full_state,
      const AttributeIESKF &_attribute_ieskf,
      const PointCloud::Ptr &_body_points)
      : full_state(_full_state), attribute_ieskf(_attribute_ieskf), body_points(_body_points){};

  FullStateMsg full_state;
  AttributeIESKF attribute_ieskf;
  PointCloud::Ptr body_points;
  DebugMsgs debug_msgs;
  std::vector<ImuMsg> imu_vec;
};

struct MeaureGroup
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int sweep_id;

  std::vector<ImuMsg> imu_vec;
  std::vector<MotorMsg> motor_vec;
  PointCloud::Ptr lidar_points;

  double pcl_start_time;
  double pcl_end_time;
};

};  // namespace lixel

POINT_CLOUD_REGISTER_POINT_STRUCT(
    lixel::PointIRT,
    (float,
     x,
     x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))