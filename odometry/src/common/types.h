#pragma once


#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>

#include "common/rigid_transform.h"

using Quaternion = Eigen::Quaterniond;
using Vector3    = Eigen::Vector3d;
using Matrix3    = Eigen::Matrix3d;


struct EIGEN_ALIGN16 PointXYZIRT {
  PCL_ADD_POINT4D;
  float intensity;
  double timestamp;
  uint16_t ring;
  PCL_MAKE_ALIGNED_OPERATOR_NEW
};

// clang-format off
POINT_CLOUD_REGISTER_POINT_STRUCT(
  PointXYZIRT,
  (float, x, x)
  (float, y, y)
  (float, z, z)
  (float, intensity, intensity)
  (double, timestamp, timestamp)
  (std::uint16_t, ring, ring)
)
// clang-format on

using PointCloud = pcl::PointCloud<PointXYZIRT>;

struct LidarMsg {
  using Ptr      = std::shared_ptr<LidarMsg>;
  using ConstPtr = std::shared_ptr<const LidarMsg>;

  PointCloud::Ptr lidar_points;

  LidarMsg() : lidar_points(new PointCloud()) {}
};

struct ImuMsg {
  using Ptr      = std::shared_ptr<ImuMsg>;
  using ConstPtr = std::shared_ptr<const ImuMsg>;

  double timestamp;
  Vector3 acc;  // in m/s^2
  Vector3 gyr;  // in rad/s
};

struct EncoderMsg {
  using Ptr      = std::shared_ptr<EncoderMsg>;
  using ConstPtr = std::shared_ptr<const EncoderMsg>;

  double timestamp;
  Quaternion rot;
};

struct MsgPack {
  using Ptr      = std::shared_ptr<MsgPack>;
  using ConstPtr = std::shared_ptr<const MsgPack>;

  int id;

  double group_start_time;
  double group_end_time;

  PointCloud::Ptr lidar_points;
  std::vector<ImuMsg> imu_msgs;
  std::vector<EncoderMsg> encoder_msgs;
};

struct OdometryResult {
  using Ptr      = std::shared_ptr<OdometryResult>;
  using ConstPtr = std::shared_ptr<const OdometryResult>;

  double timestamp;

  Quaternion rot;
  Vector3 pos;
  Vector3 vel;
  Vector3 ba;
  Vector3 bg;
  Vector3 grav;

  PointCloud::Ptr points;
};

#define SKEW_SYM_MATRX(v) 0.0,-v[2],v[1],v[2],0.0,-v[0],-v[1],v[0],0.0

template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &&ang)
{
    T ang_norm = ang.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (ang_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang / ang_norm;
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_axis);
        /// Roderigous Tranformation
        return Eye3 + std::sin(ang_norm) * K + (1.0 - std::cos(ang_norm)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

template<typename T, typename Ts>
Eigen::Matrix<T, 3, 3> Exp(const Eigen::Matrix<T, 3, 1> &ang_vel, const Ts &dt)
{
    T ang_vel_norm = ang_vel.norm();
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();

    if (ang_vel_norm > 0.0000001)
    {
        Eigen::Matrix<T, 3, 1> r_axis = ang_vel / ang_vel_norm;
        Eigen::Matrix<T, 3, 3> K;

        K << SKEW_SYM_MATRX(r_axis);

        T r_ang = ang_vel_norm * dt;

        /// Roderigous Tranformation
        return Eye3 + std::sin(r_ang) * K + (1.0 - std::cos(r_ang)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

template<typename T>
Eigen::Matrix<T, 3, 3> Exp(const T &v1, const T &v2, const T &v3)
{
    T &&norm = sqrt(v1 * v1 + v2 * v2 + v3 * v3);
    Eigen::Matrix<T, 3, 3> Eye3 = Eigen::Matrix<T, 3, 3>::Identity();
    if (norm > 0.00001)
    {
        T r_ang[3] = {v1 / norm, v2 / norm, v3 / norm};
        Eigen::Matrix<T, 3, 3> K;
        K << SKEW_SYM_MATRX(r_ang);

        /// Roderigous Tranformation
        return Eye3 + std::sin(norm) * K + (1.0 - std::cos(norm)) * K * K;
    }
    else
    {
        return Eye3;
    }
}

/* Logrithm of a Rotation Matrix */
template<typename T>
Eigen::Matrix<T,3,1> Log(const Eigen::Matrix<T, 3, 3> &R)
{
    T theta = (R.trace() > 3.0 - 1e-6) ? 0.0 : std::acos(0.5 * (R.trace() - 1));
    Eigen::Matrix<T,3,1> K(R(2,1) - R(1,2), R(0,2) - R(2,0), R(1,0) - R(0,1));
    return (std::abs(theta) < 0.001) ? (0.5 * K) : (0.5 * theta / std::sin(theta) * K);
}

template<typename T>
Eigen::Matrix<T, 3, 1> RotMtoEuler(const Eigen::Matrix<T, 3, 3> &rot)
{
    T sy = sqrt(rot(0,0)*rot(0,0) + rot(1,0)*rot(1,0));
    bool singular = sy < 1e-6;
    T x, y, z;
    if(!singular)
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
    Eigen::Matrix<T, 3, 1> ang(x, y, z);
    return ang;
}

#define DIM_STATE (18)  // Dimension of states (Let Dim(SO(3)) = 3)
#define INIT_COV (0.0000001)
#define VEC_FROM_ARRAY(v) v[0], v[1], v[2]

struct StatesGroup {
  StatesGroup() {
    this->rot_end = Matrix3::Identity();
    this->pos_end = Vector3::Zero();
    this->vel_end = Vector3::Zero();
    this->bias_g  = Vector3::Zero();
    this->bias_a  = Vector3::Zero();
    this->gravity = Vector3::Zero();
    this->cov     = Eigen::Matrix<double, DIM_STATE, DIM_STATE>::Identity() * INIT_COV;
  };

  StatesGroup(const StatesGroup &b) {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g  = b.bias_g;
    this->bias_a  = b.bias_a;
    this->gravity = b.gravity;
    this->cov     = b.cov;
  };

  StatesGroup &operator=(const StatesGroup &b) {
    this->rot_end = b.rot_end;
    this->pos_end = b.pos_end;
    this->vel_end = b.vel_end;
    this->bias_g  = b.bias_g;
    this->bias_a  = b.bias_a;
    this->gravity = b.gravity;
    this->cov     = b.cov;
    return *this;
  };

  StatesGroup operator+(const Eigen::Matrix<double, DIM_STATE, 1> &state_add) {
    StatesGroup a;
    a.rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    a.pos_end = this->pos_end + state_add.block<3, 1>(3, 0);
    a.vel_end = this->vel_end + state_add.block<3, 1>(6, 0);
    a.bias_g  = this->bias_g + state_add.block<3, 1>(9, 0);
    a.bias_a  = this->bias_a + state_add.block<3, 1>(12, 0);
    a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
    a.cov     = this->cov;
    return a;
  };

  StatesGroup &operator+=(const Eigen::Matrix<double, DIM_STATE, 1> &state_add) {
    this->rot_end = this->rot_end * Exp(state_add(0, 0), state_add(1, 0), state_add(2, 0));
    this->pos_end += state_add.block<3, 1>(3, 0);
    this->vel_end += state_add.block<3, 1>(6, 0);
    this->bias_g += state_add.block<3, 1>(9, 0);
    this->bias_a += state_add.block<3, 1>(12, 0);
    this->gravity += state_add.block<3, 1>(15, 0);
    return *this;
  };

  Eigen::Matrix<double, DIM_STATE, 1> operator-(const StatesGroup &b) {
    Eigen::Matrix<double, DIM_STATE, 1> a;
    Matrix3 rotd(b.rot_end.transpose() * this->rot_end);
    a.block<3, 1>(0, 0)  = Log(rotd);
    a.block<3, 1>(3, 0)  = this->pos_end - b.pos_end;
    a.block<3, 1>(6, 0)  = this->vel_end - b.vel_end;
    a.block<3, 1>(9, 0)  = this->bias_g - b.bias_g;
    a.block<3, 1>(12, 0) = this->bias_a - b.bias_a;
    a.block<3, 1>(15, 0) = this->gravity - b.gravity;
    return a;
  };

  void resetpose() {
    this->rot_end = Matrix3::Identity();
    this->pos_end = Vector3::Zero();
    this->vel_end = Vector3::Zero();
  }

  Matrix3 rot_end;                                  // the estimated attitude (rotation matrix) at the end lidar
                                                    // point
  Vector3 pos_end;                                  // the estimated position at the end lidar point (world frame)
  Vector3 vel_end;                                  // the estimated velocity at the end lidar point (world frame)
  Vector3 bias_g;                                   // gyroscope bias
  Vector3 bias_a;                                   // accelerator bias
  Vector3 gravity;                                  // the estimated gravity acceleration
  Eigen::Matrix<double, DIM_STATE, DIM_STATE> cov;  // states covariance
};


// IMU Intrinsic
class ImuInstrinsic {
 public:
  virtual void Deskew(Eigen::Vector3d& acc, Eigen::Vector3d& gyr) const = 0;
};

// IMU Intrinsic TPM ICRA 2014 Model
class ImuInstrinsicTpmIcra2014 : public ImuInstrinsic {
 public:
  Matrix3 Ta;
  Matrix3 Ka;
  Vector3 Ba;

  Matrix3 Tg;
  Matrix3 Kg;
  Vector3 Bg;

  ImuInstrinsicTpmIcra2014(const Matrix3& Ta, const Matrix3& Ka, const Vector3& Ba, const Matrix3& Tg, const Matrix3& Kg, const Vector3& Bg)
      : Ta(Ta), Ka(Ka), Ba(Ba), Tg(Tg), Kg(Kg), Bg(Bg) {}

  virtual void Deskew(Eigen::Vector3d& acc, Eigen::Vector3d& gyr) const override {
    acc = Ta * Ka * (acc - Ba);  // different from the paper
    gyr = Tg * Kg * (gyr - Bg);
  }
};

// Lidar Intrinsic
class LidarInstrinsic {
 public:
  virtual void Deskew(Eigen::Map<Eigen::Vector3f>& point) const = 0;
};

class LidarInstrinsicSimple : public LidarInstrinsic {
 public:
  double elevation_corr;
  Eigen::Vector2d s;

  LidarInstrinsicSimple() = default;

  LidarInstrinsicSimple(double e, const Eigen::Vector2d& s) : elevation_corr(e), s(s) {}

  virtual void Deskew(Eigen::Map<Eigen::Vector3f>& point) const override {
    // when point is on z-axis
    if (point.head<2>().norm() < 1e-7) {
      return;
    }

    const double norm  = point.norm();
    const double theta = std::atan2(point.y(), point.x());
    const double phi   = std::asin(point.z() / norm) + elevation_corr;

    point.x() = float(norm * std::cos(phi) * std::cos(theta));
    point.y() = float(norm * std::cos(phi) * std::sin(theta));
    point.z() = float(norm * std::sin(phi));
  }
};

// Sensor Calibration
class SensorCalib {
 public:
  bool has_encoder;  // if false, encoder_to_imu will be ignored, only lidar_to_encoder will be used
  Rigid3d lidar_to_encoder;
  Rigid3d encoder_to_imu;
  std::shared_ptr<LidarInstrinsic> lidar_instrinsic;
  std::shared_ptr<ImuInstrinsic> imu_instrinsic;

  SensorCalib(bool encoder, const Rigid3d& l2e, const Rigid3d& e2i, const std::shared_ptr<LidarInstrinsic>& lidar_instrinsic,
              const std::shared_ptr<ImuInstrinsic>& imu_instrinsic)
      : has_encoder(encoder), lidar_to_encoder(l2e), encoder_to_imu(e2i), lidar_instrinsic(lidar_instrinsic), imu_instrinsic(imu_instrinsic) {}
};
