#pragma once

#define EIGEN_STACK_ALLOCATION_LIMIT 10048576
#include <Eigen/Core>
#include <Eigen/Dense>

#include "common/common_struct.h"
#include "common/math_utils.h"
#include "lio_msgs.h"
#include "parameters.h"
#include "sensor_fusion/base_fusion.h"
#include "state_group.h"
using namespace Eigen;

namespace lixel
{
constexpr double CONVERAGE_ROT_THRESHOLD = 5e-5;    // unit: rad/s
constexpr double CONVERAGE_TRANS_THRESHOLD = 5e-4;  // unit: m
constexpr double SCALE = 1e6;

#if GRAVITY_CALIBRATION
constexpr double INIT_GRAVITY = 1e-6;  // 1e-6 m/s^2
#endif
enum PredictMethod
{
  MID_POINT,
  DOUBLE_SAMPLING
};

struct IESKFConfigs
{
  Vec3 mill_cov_acc;
  Vec3 mill_cov_gyr;
  Vec3 mill_cov_bias_gyr;
  Vec3 mill_cov_bias_acc;
  Vec3 gravity;
  int max_iter_num;
  double acc_keep_std_limit;
  double gyro_keep_std_limit;
  PredictMethod predict_method;
};
// TODO: add state machine about IESKF (Init, Run, Failed, Save, ...)
class IESKF
{
 private:
  KFState::Ptr states_ptr_;
  std::vector<ImuMsg> static_imu_vec_;
  Vec3 last_dtheta_ = Vec3::Zero();
  Vec3 last_dv_ = Vec3::Zero();
  IESKFConfigs ieskf_configs_;

  bool init_ = false;
  double last_pcl_end_time_ = 0.0;
  double registration_ts_ = 0.0;
  double matrix_calculation_ts_ = 0.0;
  double init_ts_ = 0.0;

  void propagateState(const Vec3& hat_gyro, const Vec3& hat_acc, double dt, AttributeImu& attr_imu);
  void propagateCov(const Vec3& hat_gyro, const Vec3& hat_acc, double dt);
  void stateSliding();

 public:
  using Ptr = std::shared_ptr<IESKF>;
  using ConstPtr = std::shared_ptr<const IESKF>;

  explicit IESKF(const IESKFParam& param);

  void init(const KFState& init_state);
  void predict(
      const std::vector<ImuMsg>& imu_vec,
      double pcl_end_time,
      int sweep_id,
      StatePredict& state_predict,
      AttributePredict& attr_predict);
  void update(BaseFusion& fusion, AttributeIterate& attr_iter);
  void logState(std::string str);
  KFState::ConstPtr getStatesPtr() const;

  ~IESKF();
};
}  // namespace lixel
