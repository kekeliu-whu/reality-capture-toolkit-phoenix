#include "ieskf/ieskf.h"

namespace lixel
{
IESKF::IESKF(const IESKFParam &param)
{
  states_ptr_ = std::make_shared<KFState>(param.window_size);

  ieskf_configs_.mill_cov_acc =
      static_cast<FloatDataType>(pow(param.acc_std, 2) * SCALE) * Vec3::Ones();
  ieskf_configs_.mill_cov_gyr =
      static_cast<FloatDataType>(pow(param.gyr_std, 2) * SCALE) * Vec3::Ones();
  ieskf_configs_.mill_cov_bias_gyr =
      static_cast<FloatDataType>(pow(param.gyr_bias_std, 2) * SCALE) * Vec3::Ones();
  ieskf_configs_.mill_cov_bias_acc =
      static_cast<FloatDataType>(pow(param.acc_bias_std, 2) * SCALE) * Vec3::Ones();
  ieskf_configs_.acc_std = param.acc_std;
  ieskf_configs_.gyr_std = param.gyr_std;
  ieskf_configs_.acc_std_slope = param.acc_std_slope;
  ieskf_configs_.gyro_std_slope = param.gyro_std_slope;
  ieskf_configs_.lidar_variance_limit =
      param.lidar_std_dev_limit * param.lidar_std_dev_limit;
  ieskf_configs_.gravity = DEFAULT_GRIVITY_VEC;
  ieskf_configs_.max_iter_num = param.max_iter;
  // The production DLL does not expose this as a configuration option.
  ieskf_configs_.predict_method = DOUBLE_SAMPLING;
  init_ = false;

  if (states_ptr_->windowSize() < 1)
  {
    LOG(ERROR) << "STATE WINDOW SIZE <= 1 !!!! SOMETHING WRONG";
    exit(0);
  }
  LOG(INFO) << "IESKF configured window_size=" << states_ptr_->windowSize()
            << " state_dim=" << states_ptr_->dimState()
            << " scalar_bytes=" << sizeof(FloatDataType)
            << " predict_method=DOUBLE_SAMPLING";
}

IESKF::~IESKF()
{
}

KFState::ConstPtr IESKF::getStatesPtr() const
{
  return states_ptr_;
}

void IESKF::logState(std::string str)
{
  Vec3 euler = rotMtoEuler(states_ptr_->sw_rot_[0]) * RAD_2_DEG;
  LOG(INFO) << str + "euler_x:" << euler.x();
  LOG(INFO) << str + "euler_y:" << euler.y();
  LOG(INFO) << str + "euler_z:" << euler.z();
  LOG(INFO) << str + "pos_x:" << states_ptr_->sw_pos_[0].x();
  LOG(INFO) << str + "pos_y:" << states_ptr_->sw_pos_[0].y();
  LOG(INFO) << str + "pos_z:" << states_ptr_->sw_pos_[0].z();
  LOG(INFO) << str + "vel_x:" << states_ptr_->vel_.x();
  LOG(INFO) << str + "vel_y:" << states_ptr_->vel_.y();
  LOG(INFO) << str + "vel_z:" << states_ptr_->vel_.z();
  LOG(INFO) << str + "ba_x:" << states_ptr_->acc_bias_.x();
  LOG(INFO) << str + "ba_y:" << states_ptr_->acc_bias_.y();
  LOG(INFO) << str + "ba_z:" << states_ptr_->acc_bias_.z();
  LOG(INFO) << str + "bg_x:" << states_ptr_->gyo_bias_.x();
  LOG(INFO) << str + "bg_y:" << states_ptr_->gyo_bias_.y();
  LOG(INFO) << str + "bg_z:" << states_ptr_->gyo_bias_.z();

  LOG(INFO) << str + "std_rot_x:" << sqrt(states_ptr_->mill_cov(0, 0));
  LOG(INFO) << str + "std_rot_y:" << sqrt(states_ptr_->mill_cov(1, 1));
  LOG(INFO) << str + "std_rot_z:" << sqrt(states_ptr_->mill_cov(2, 2));
  LOG(INFO) << str + "std_pos_x:" << sqrt(states_ptr_->mill_cov(3, 3));
  LOG(INFO) << str + "std_pos_y:" << sqrt(states_ptr_->mill_cov(4, 4));
  LOG(INFO) << str + "std_pos_z:" << sqrt(states_ptr_->mill_cov(5, 5));
  LOG(INFO) << str + "std_vel_x:" << sqrt(states_ptr_->mill_cov(6, 6));
  LOG(INFO) << str + "std_vel_y:" << sqrt(states_ptr_->mill_cov(7, 7));
  LOG(INFO) << str + "std_vel_z:" << sqrt(states_ptr_->mill_cov(8, 8));
  LOG(INFO) << str + "std_bg_x:" << sqrt(states_ptr_->mill_cov(9, 9));
  LOG(INFO) << str + "std_bg_y:" << sqrt(states_ptr_->mill_cov(10, 10));
  LOG(INFO) << str + "std_bg_z:" << sqrt(states_ptr_->mill_cov(11, 11));
  LOG(INFO) << str + "std_ba_x:" << sqrt(states_ptr_->mill_cov(12, 12));
  LOG(INFO) << str + "std_ba_y:" << sqrt(states_ptr_->mill_cov(13, 13));
  LOG(INFO) << str + "std_ba_z:" << sqrt(states_ptr_->mill_cov(14, 14));

#if GRAVITY_CALIBRATION
  LOG(INFO) << "gravity_x:" << states_ptr_->gravity.x();
  LOG(INFO) << "gravity_y:" << states_ptr_->gravity.y();
  LOG(INFO) << "gravity_z:" << states_ptr_->gravity.z();
  LOG(INFO) << "std_gravity_x:" << sqrt(states_ptr_->cov(15, 15));
  LOG(INFO) << "std_gravity_y:" << sqrt(states_ptr_->cov(16, 16));
  LOG(INFO) << "std_gravity_z:" << sqrt(states_ptr_->cov(17, 17));
#endif

  if (str == "update_")
  {
    LOG(INFO) << "registrationT:" << registration_ts_;
    LOG(INFO) << "matrixT:" << matrix_calculation_ts_;
    matrix_calculation_ts_ = 0;
    registration_ts_ = 0;
  }
}

void IESKF::init(const KFState &init_state)
{
  if (init_state.windowSize() != states_ptr_->windowSize())
    throw std::invalid_argument("Initial state window size does not match IESKF configuration");
  states_ptr_ = std::make_shared<KFState>(init_state);
  last_pcl_end_time_ = init_state.timestamp;
  // Coning/sculling increments must never cross an initialization boundary.
  last_dtheta_.setZero();
  last_dv_.setZero();
  init_ = true;
  logState("init");
}

}  // namespace lixel
