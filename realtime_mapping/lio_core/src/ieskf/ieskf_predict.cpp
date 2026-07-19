#include "ieskf/ieskf.h"

namespace
{
inline void interpolateIMU(
    const lixel::ImuMsg& imu_start,
    const lixel::ImuMsg& imu_end,
    lixel::ImuMsg& imu_target_time,
    double target_time)
{
  double factor = (target_time - imu_start.timestamp) / (imu_end.timestamp - imu_start.timestamp);
  imu_target_time.timestamp = target_time;
  imu_target_time.gyro = factor * imu_end.gyro + (1 - factor) * imu_start.gyro;
  imu_target_time.acc = factor * imu_end.acc + (1 - factor) * imu_start.acc;
}

std::vector<lixel::ImuMsg>
filterIMU(const std::vector<lixel::ImuMsg>& imu_vec, double pcl_end_time, double last_pcl_end_time)
{
  std::vector<lixel::ImuMsg> filter_imu_vec;
  int len = static_cast<int>(imu_vec.size());
  for (int i = 0; i < len - 1; ++i)
  {
    const lixel::ImuMsg& head = imu_vec.at(i);
    const lixel::ImuMsg& tail = imu_vec.at(i + 1);

    lixel::ImuMsg imu_msg;
    if (head.timestamp < last_pcl_end_time)
    {
      interpolateIMU(head, tail, imu_msg, last_pcl_end_time);
      filter_imu_vec.push_back(imu_msg);
    }
    else if (head.timestamp < pcl_end_time)
    {
      filter_imu_vec.push_back(head);
    }

    if (head.timestamp < pcl_end_time && tail.timestamp >= pcl_end_time)
    {
      interpolateIMU(head, tail, imu_msg, pcl_end_time);
      filter_imu_vec.push_back(imu_msg);
      break;
    }
  }
  return filter_imu_vec;
}
}  // namespace

namespace lixel
{
void IESKF::predict(
    const std::vector<ImuMsg>& imu_vec,
    double pcl_end_time,
    int sweep_id,
    StatePredict& state_predict,
    AttributePredict& attr_predict)
{
  state_predict.clear();
  attr_predict.clear();
  /// CHECK
  if (imu_vec.size() <= 2 || !init_)
  {
    lslog(LSLOG_ERROR) << "KF not init or imu size <= 2";
    return;
  }

  if (pcl_end_time <= last_pcl_end_time_)
  {
    lslog(LSLOG_ERROR) << "now pcl_end_time is less than last pcl_end_time";
    return;
  }

  lslog(LSLOG_INFO) << std::fixed << "last_pcl_end_time_:" << last_pcl_end_time_;
  lslog(LSLOG_INFO) << std::fixed << "pcl_end_time:" << pcl_end_time;

  lslog(LSLOG_INFO) << "imu_vec.size:" << imu_vec.size();
  const std::vector<ImuMsg>& filtered_imu_vec = ::filterIMU(imu_vec, pcl_end_time, last_pcl_end_time_);
  lslog(LSLOG_INFO) << "filtered_imu_vec.size:" << filtered_imu_vec.size();

  Vec3 acc_mean, gyro_mean, acc_world;
  double dt, now_iter_ts;
  PosAtt first_pos;
  first_pos.timestamp = states_ptr_->timestamp;
  first_pos.pos = states_ptr_->sw_pos_[0].cast<float>();
  first_pos.quat = QUATD(states_ptr_->sw_rot_[0]).cast<float>();
  state_predict.emplace_back(first_pos);
  lslog(LSLOG_INFO) << std::fixed << "first_pos:" << first_pos.timestamp;
  stateSliding();

  for (int i = 0; i < filtered_imu_vec.size() - 1; ++i)
  {
    const ImuMsg& head = filtered_imu_vec.at(i);
    const ImuMsg& tail = filtered_imu_vec.at(i + 1);

    gyro_mean = ((head.gyro + tail.gyro) / 2).cast<FloatDataType>();
    acc_mean = ((head.acc + tail.acc) / 2).cast<FloatDataType>();
    gyro_mean -= states_ptr_->gyo_bias_;
    acc_mean -= states_ptr_->acc_bias_;
    now_iter_ts = tail.timestamp;
    dt = tail.timestamp - head.timestamp;

    AttributeImu attr_imu;
    attr_imu.sweep_id = sweep_id;
    attr_imu.timestamp = now_iter_ts;
    /*** covariance propagation of error state ***/
    propagateCov(gyro_mean, acc_mean, dt);
    /*** propogation of nonmial state ***/
    propagateState(gyro_mean, acc_mean, dt, attr_imu);

    attr_predict.push_back(attr_imu);

    /*** Imu Frequences Poses ***/
    PosAtt now_iter_pos;
    now_iter_pos.timestamp = now_iter_ts;
    now_iter_pos.pos = states_ptr_->sw_pos_[0].cast<float>();
    now_iter_pos.quat = QUATD(states_ptr_->sw_rot_[0]).cast<float>();
    state_predict.emplace_back(now_iter_pos);
  }
  states_ptr_->timestamp = pcl_end_time;
  // timestamp sliding
  for (int i = WINDOW_SIZE - 1; i >= 1; --i)
  {
    states_ptr_->sw_timestamp[i] = states_ptr_->sw_timestamp[i - 1];
  }
  states_ptr_->sw_timestamp[0] = pcl_end_time;

  last_pcl_end_time_ = pcl_end_time;

  // logState("predict_");
}

void IESKF::propagateState(const Vec3& hat_gyro, const Vec3& hat_acc, const double dt, AttributeImu& attr_imu)
{
  Vec3 gravity = ieskf_configs_.gravity;
#if GRAVITY_CALIBRATION
  gravity = states_ptr_->gravity;
#endif
  Vec3 acc_world_plus_gravity = states_ptr_->sw_rot_[0] * hat_acc + gravity;

  Vec3 d_theta = hat_gyro * dt;
  Vec3 d_v = hat_acc * dt;
  Vec3 d_v_temp = d_v + 0.5 * d_theta.cross(d_v) + (last_dtheta_.cross(d_v) + last_dv_.cross(d_theta)) / 12;

  Vec3 pre_vel = states_ptr_->vel_;

  Vec3& vel = states_ptr_->vel_;
  Vec3& pos = states_ptr_->sw_pos_[0];
  Mat3& rot = states_ptr_->sw_rot_[0];
  switch (ieskf_configs_.predict_method)
  {
    case DOUBLE_SAMPLING:
      // TODO: DOUBLE_SAMPLING have bug !!!!!!
      vel = vel + (rot * d_v_temp) + gravity * dt;
      pos = pos + (vel + pre_vel) / 2 * dt;
      rot = rot * exp((d_theta + last_dtheta_.cross(d_theta) / 12));
      last_dv_ = d_v;
      last_dtheta_ = d_theta;
      break;
    case MID_POINT:
      pos = pos + vel * dt + 0.5 * acc_world_plus_gravity * dt * dt;
      vel = vel + acc_world_plus_gravity * dt;
      rot = rot * exp(d_theta);
      break;
    default:
      break;
  }
  attr_imu.dt = static_cast<float>(dt);
  attr_imu.gyro_true = hat_gyro.cast<float>();
  attr_imu.acc_true = hat_acc.cast<float>();
  attr_imu.gyro_world = (rot * hat_gyro).cast<float>();
  attr_imu.acc_world = acc_world_plus_gravity.cast<float>();
}

void IESKF::propagateCov(const Vec3& hat_gyro, const Vec3& hat_acc, const double dt)
{
  Mat3 acc_avr_skew;
  acc_avr_skew << skewSymMatrix(hat_acc);
  Mat3& rot = states_ptr_->sw_rot_[0];
  MatDIM& cov = states_ptr_->mill_cov;

  /*** error state order: dtheta, dt, dv, dbg, dba ***/
  MatDIM F_x, Q;
  F_x.setIdentity();
  F_x.block<3, 3>(0, 0) = exp(hat_gyro * -dt);
  F_x.block<3, 3>(0, 9) = -Mat3::Identity() * dt;
  F_x.block<3, 3>(3, 6) = Mat3::Identity() * dt;
  F_x.block<3, 3>(6, 0) = -rot * acc_avr_skew * dt;
  F_x.block<3, 3>(6, 12) = -rot * dt;
#if GRAVITY_CALIBRATION
  F_x.block<3, 3>(6, 15) = Mat3::Identity() * dt;
#endif
  double gyro_radio = hat_gyro.norm() / ieskf_configs_.gyro_keep_std_limit;
  gyro_radio = gyro_radio < 1.0 ? 1.0 : gyro_radio;
  gyro_radio = gyro_radio > 2.0 ? 2.0 : gyro_radio;
  double acc_radio =
      (states_ptr_->sw_rot_[0] * hat_acc + ieskf_configs_.gravity).norm() / ieskf_configs_.acc_keep_std_limit;
  acc_radio = acc_radio < 1.0 ? 1.0 : acc_radio;
  acc_radio = acc_radio > 2.0 ? 2.0 : acc_radio;

  Q.setZero();
  Q.block<3, 3>(0, 0).diagonal() = ieskf_configs_.mill_cov_gyr * std::pow(gyro_radio, 2) * dt * dt;
  Q.block<3, 3>(6, 6) =
      rot * ieskf_configs_.mill_cov_acc.asDiagonal() * rot.transpose() * std::pow(acc_radio, 2) * dt * dt;
  Q.block<3, 3>(9, 9).diagonal() = ieskf_configs_.mill_cov_bias_gyr * dt * dt;    // bias gyro covariance
  Q.block<3, 3>(12, 12).diagonal() = ieskf_configs_.mill_cov_bias_acc * dt * dt;  // bias acc covariance
  // P = FPF^T + Q
  SparseMat F_x_sparse = F_x.sparseView();
  cov = F_x_sparse * cov * F_x_sparse.transpose() + Q;
}

void IESKF::stateSliding()
{
  if (WINDOW_SIZE <= 1)
    return;

  // 1. state sliding
  for (int i = WINDOW_SIZE - 1; i >= 1; --i)
  {
    states_ptr_->sw_pos_[i] = states_ptr_->sw_pos_[i - 1];
    states_ptr_->sw_rot_[i] = states_ptr_->sw_rot_[i - 1];
  }

  // 2. covariance maintain
  MatDIM F = MatDIM::Zero();
  F.template block<DIM_CURR_STATE, DIM_CURR_STATE>(0, 0) =
      Matrix<FloatDataType, DIM_CURR_STATE, DIM_CURR_STATE>::Identity();
  for (int i = 2; i < WINDOW_SIZE; ++i)
  {
    int dim = DIM_CURR_STATE + (i - 1) * 6;
    F.template block<3, 3>(dim, dim - 6) = M3D::Identity();
    F.template block<3, 3>(dim + 3, dim - 3) = M3D::Identity();
  }
  F.template block<3, 3>(DIM_CURR_STATE, 0) = M3D::Identity();
  F.template block<3, 3>(DIM_CURR_STATE + 3, 3) = M3D::Identity();
  SparseMat F_sparse = F.sparseView();
  states_ptr_->mill_cov = F_sparse * states_ptr_->mill_cov * F_sparse.transpose();
}

}  // namespace lixel
