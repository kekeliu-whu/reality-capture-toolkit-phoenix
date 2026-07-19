#include "ieskf/ieskf.h"

namespace lixel
{
IESKF::IESKF(const IESKFParam &param)
{
  states_ptr_ = std::make_shared<KFState>();

  ieskf_configs_.mill_cov_acc = pow(param.acc_std, 2) * Vec3::Ones() * SCALE;
  ieskf_configs_.mill_cov_gyr = pow(param.gyr_std, 2) * Vec3::Ones() * SCALE;
  ieskf_configs_.mill_cov_bias_gyr = pow(param.gyr_bias_std, 2) * Vec3::Ones() * SCALE;
  ieskf_configs_.mill_cov_bias_acc = pow(param.acc_bias_std, 2) * Vec3::Ones() * SCALE;
  ieskf_configs_.acc_keep_std_limit = param.acc_keep_std_limit;
  ieskf_configs_.gyro_keep_std_limit = param.gyro_keep_std_limit;
  ieskf_configs_.gravity = DEFAULT_GRIVITY_VEC;
  ieskf_configs_.max_iter_num = param.max_iter;
  ieskf_configs_.predict_method = DOUBLE_SAMPLING;
  init_ = false;

  if (WINDOW_SIZE < 1)
  {
    lslog(LSLOG_ERROR) << "STATE WINDOW SIZE <= 1 !!!! SOMETHING WRONG";
    exit(0);
  }
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
  lslog(LSLOG_INFO) << str + "euler_x:" << euler.x();
  lslog(LSLOG_INFO) << str + "euler_y:" << euler.y();
  lslog(LSLOG_INFO) << str + "euler_z:" << euler.z();
  lslog(LSLOG_INFO) << str + "pos_x:" << states_ptr_->sw_pos_[0].x();
  lslog(LSLOG_INFO) << str + "pos_y:" << states_ptr_->sw_pos_[0].y();
  lslog(LSLOG_INFO) << str + "pos_z:" << states_ptr_->sw_pos_[0].z();
  lslog(LSLOG_INFO) << str + "vel_x:" << states_ptr_->vel_.x();
  lslog(LSLOG_INFO) << str + "vel_y:" << states_ptr_->vel_.y();
  lslog(LSLOG_INFO) << str + "vel_z:" << states_ptr_->vel_.z();
  lslog(LSLOG_INFO) << str + "ba_x:" << states_ptr_->acc_bias_.x();
  lslog(LSLOG_INFO) << str + "ba_y:" << states_ptr_->acc_bias_.y();
  lslog(LSLOG_INFO) << str + "ba_z:" << states_ptr_->acc_bias_.z();
  lslog(LSLOG_INFO) << str + "bg_x:" << states_ptr_->gyo_bias_.x();
  lslog(LSLOG_INFO) << str + "bg_y:" << states_ptr_->gyo_bias_.y();
  lslog(LSLOG_INFO) << str + "bg_z:" << states_ptr_->gyo_bias_.z();

  lslog(LSLOG_INFO) << str + "std_rot_x:" << sqrt(states_ptr_->mill_cov(0, 0));
  lslog(LSLOG_INFO) << str + "std_rot_y:" << sqrt(states_ptr_->mill_cov(1, 1));
  lslog(LSLOG_INFO) << str + "std_rot_z:" << sqrt(states_ptr_->mill_cov(2, 2));
  lslog(LSLOG_INFO) << str + "std_pos_x:" << sqrt(states_ptr_->mill_cov(3, 3));
  lslog(LSLOG_INFO) << str + "std_pos_y:" << sqrt(states_ptr_->mill_cov(4, 4));
  lslog(LSLOG_INFO) << str + "std_pos_z:" << sqrt(states_ptr_->mill_cov(5, 5));
  lslog(LSLOG_INFO) << str + "std_vel_x:" << sqrt(states_ptr_->mill_cov(6, 6));
  lslog(LSLOG_INFO) << str + "std_vel_y:" << sqrt(states_ptr_->mill_cov(7, 7));
  lslog(LSLOG_INFO) << str + "std_vel_z:" << sqrt(states_ptr_->mill_cov(8, 8));
  lslog(LSLOG_INFO) << str + "std_bg_x:" << sqrt(states_ptr_->mill_cov(9, 9));
  lslog(LSLOG_INFO) << str + "std_bg_y:" << sqrt(states_ptr_->mill_cov(10, 10));
  lslog(LSLOG_INFO) << str + "std_bg_z:" << sqrt(states_ptr_->mill_cov(11, 11));
  lslog(LSLOG_INFO) << str + "std_ba_x:" << sqrt(states_ptr_->mill_cov(12, 12));
  lslog(LSLOG_INFO) << str + "std_ba_y:" << sqrt(states_ptr_->mill_cov(13, 13));
  lslog(LSLOG_INFO) << str + "std_ba_z:" << sqrt(states_ptr_->mill_cov(14, 14));

#if GRAVITY_CALIBRATION
  lslog(LSLOG_INFO) << "gravity_x:" << states_ptr_->gravity.x();
  lslog(LSLOG_INFO) << "gravity_y:" << states_ptr_->gravity.y();
  lslog(LSLOG_INFO) << "gravity_z:" << states_ptr_->gravity.z();
  lslog(LSLOG_INFO) << "std_gravity_x:" << sqrt(states_ptr_->cov(15, 15));
  lslog(LSLOG_INFO) << "std_gravity_y:" << sqrt(states_ptr_->cov(16, 16));
  lslog(LSLOG_INFO) << "std_gravity_z:" << sqrt(states_ptr_->cov(17, 17));
#endif

  if (str == "update_")
  {
    lslog(LSLOG_INFO) << "registrationT:" << registration_ts_;
    lslog(LSLOG_INFO) << "matrixT:" << matrix_calculation_ts_;
    matrix_calculation_ts_ = 0;
    registration_ts_ = 0;
  }
}

void IESKF::init(const KFState &init_state)
{
  states_ptr_ = std::make_shared<KFState>(init_state);
  last_pcl_end_time_ = init_state.timestamp;
  init_ = true;
  logState("init");
}

}  // namespace lixel
