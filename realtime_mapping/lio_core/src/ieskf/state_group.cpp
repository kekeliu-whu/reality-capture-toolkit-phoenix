//
// Created by youyuan on 24-2-4.
//

#include "ieskf/state_group.h"

using namespace lixel;

KFState::KFState()
{
  this->timestamp = 0;
  this->sw_pos_[0] = Vec3::Zero();
  this->sw_rot_[0] = Mat3::Identity();
  this->vel_ = Vec3::Zero();
  this->gyo_bias_ = Vec3::Zero();
  this->acc_bias_ = Vec3::Zero();

  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    this->sw_pos_[i] = Vec3::Zero();
    this->sw_rot_[i] = Mat3::Identity();
  }
#if GRAVITY_CALIBRATION
  this->gravity = Vec3::Zero();
#endif

#if EXTRINSIC_CALIBRATION
  this->R_I_L = M3F::Identity();
  this->t_I_L = V3F::Zero();
#endif

  this->mill_cov = MatDIM::Identity() * INIT_COV;
}

KFState::KFState(const lixel::KFState& b)
{
  this->timestamp = b.timestamp;
  this->sw_pos_[0] = b.sw_pos_[0];
  this->sw_rot_[0] = b.sw_rot_[0];
  this->vel_ = b.vel_;
  this->gyo_bias_ = b.gyo_bias_;
  this->acc_bias_ = b.acc_bias_;

#if GRAVITY_CALIBRATION
  this->gravity = b.gravity;
#endif

#if EXTRINSIC_CALIBRATION
  this->R_I_L = b.R_I_L;
  this->t_I_L = b.t_I_L;
#endif

  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    this->sw_pos_[i] = b.sw_pos_[i];
    this->sw_rot_[i] = b.sw_rot_[i];
  }

  this->mill_cov = b.mill_cov;
}

void KFState::resetPVQ()
{
  this->sw_pos_[0] = Vec3::Zero();
  this->sw_rot_[0] = Mat3::Identity();
  this->vel_ = Vec3::Zero();
  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    this->sw_pos_[i] = Vec3::Zero();
    this->sw_rot_[i] = Mat3::Identity();
  }
}

KFState KFState::operator+(const VecDIM& state_add)
{
  KFState a;
  a.sw_rot_[0] = this->sw_rot_[0] * exp(state_add.block<3, 1>(0, 0));
  a.sw_pos_[0] = this->sw_pos_[0] + state_add.block<3, 1>(3, 0);
  a.vel_ = this->vel_ + state_add.block<3, 1>(6, 0);
  a.gyo_bias_ = this->gyo_bias_ + state_add.block<3, 1>(9, 0);
  a.acc_bias_ = this->acc_bias_ + state_add.block<3, 1>(12, 0);

#if GRAVITY_CALIBRATION
  a.gravity = this->gravity + state_add.block<3, 1>(15, 0);
#endif

#if EXTRINSIC_CALIBRATION
  a.R_W_I = this->R_W_I * exp(state_add.block<3, 1>(18, 0));
  a.t_W_I = this->t_W_I + state_add.block<3, 1>(21, 0);
#endif

  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    int dim = DIM_CURR_STATE + (i - 1) * 6;
    a.sw_rot_[i] = this->sw_rot_[i] * exp(state_add.block<3, 1>(dim, 0));
    a.sw_pos_[i] = this->sw_pos_[i] + state_add.block<3, 1>(dim + 3, 0);
  }

  a.mill_cov = this->mill_cov;
  return a;
}

KFState& KFState::operator+=(const VecDIM& state_add)
{
  this->sw_rot_[0] = this->sw_rot_[0] * exp(state_add.block<3, 1>(0, 0));
  this->sw_pos_[0] += state_add.block<3, 1>(3, 0);
  this->vel_ += state_add.block<3, 1>(6, 0);
  this->gyo_bias_ += state_add.block<3, 1>(9, 0);
  this->acc_bias_ += state_add.block<3, 1>(12, 0);

#if GRAVITY_CALIBRATION
  this->gravity += state_add.block<3, 1>(15, 0);
#endif

#if EXTRINSIC_CALIBRATION
  this->R_W_I = this->R_W_I * exp(state_add.block<3, 1>(18, 0));
  this->t_W_I = this->t_W_I + state_add.block<3, 1>(21, 0);
#endif

  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    int dim = DIM_CURR_STATE + (i - 1) * 6;
    this->sw_rot_[i] = this->sw_rot_[i] * exp(state_add.block<3, 1>(dim, 0));
    this->sw_pos_[i] = this->sw_pos_[i] + state_add.block<3, 1>(dim + 3, 0);
  }

  return *this;
}

VecDIM KFState::operator-(const KFState& b)
{
  VecDIM a;
  Mat3 rot_pose(b.sw_rot_[0].transpose() * this->sw_rot_[0]);
  a.block<3, 1>(0, 0) = log(rot_pose);
  a.block<3, 1>(3, 0) = this->sw_pos_[0] - b.sw_pos_[0];
  a.block<3, 1>(6, 0) = this->vel_ - b.vel_;
  a.block<3, 1>(9, 0) = this->gyo_bias_ - b.gyo_bias_;
  a.block<3, 1>(12, 0) = this->acc_bias_ - b.acc_bias_;

#if GRAVITY_CALIBRATION
  a.block<3, 1>(15, 0) = this->gravity - b.gravity;
#endif

#if EXTRINSIC_CALIBRATION
  M3F rot_extrinsic(b.R_I_L.transpose() * this->R_I_L);
  a.block<3, 1>(18, 0) = log(rot_extrinsic);
  a.block<3, 1>(21, 0) = this->t_I_L - b.t_I_L;
#endif

  for (int i = 1; i < WINDOW_SIZE; ++i)
  {
    int dim = DIM_CURR_STATE + (i - 1) * 6;
    Mat3 rot_pose_history(b.sw_rot_[i].transpose() * this->sw_rot_[i]);
    a.block<3, 1>(dim, 0) = log(rot_pose_history);
    a.block<3, 1>(dim + 3, 0) = this->sw_pos_[i] - b.sw_pos_[i];
  }
  return a;
}
