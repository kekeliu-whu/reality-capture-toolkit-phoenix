//
// Created by youyuan on 24-2-4.
//

#pragma once

#define GRAVITY_CALIBRATION false
#define EXTRINSIC_CALIBRATION false

// TODO: add WINDOW_SIZE to configs !!!
#define WINDOW_SIZE 5

#include "common/common_struct.h"
#include "common/math_utils.h"
namespace lixel
{

// clang-format off
constexpr int getDIMCurrState(){
  if (GRAVITY_CALIBRATION && EXTRINSIC_CALIBRATION) return 24;
  if (GRAVITY_CALIBRATION) return 18;
  if (EXTRINSIC_CALIBRATION) return 21;
  return 15;
}
constexpr int getDIMPrevState(){
  return 6 * (WINDOW_SIZE - 1);
}

constexpr int DIM_CURR_STATE = getDIMCurrState();
constexpr int DIM_PREV_STATE = getDIMPrevState();
constexpr int DIM_STATE = DIM_CURR_STATE + DIM_PREV_STATE;
constexpr double INIT_COV = 1e-4;

using VecDIM = Eigen::Matrix<FloatDataType, DIM_STATE, 1>;
using MatDIM = Eigen::Matrix<FloatDataType, DIM_STATE, DIM_STATE>;

using VecX = Eigen::Matrix<FloatDataType, Eigen::Dynamic, 1>;
using MatXDIM = Eigen::Matrix<FloatDataType, Eigen::Dynamic, DIM_STATE>;
using MatDIMX = Eigen::Matrix<FloatDataType, DIM_STATE, Eigen::Dynamic>;
using SparseMat = Eigen::SparseMatrix<FloatDataType, Eigen::RowMajor>;
// clang-format on

/*** State Order: sw_rot***/
class KFState
{
 public:
  using Ptr = std::shared_ptr<KFState>;
  using ConstPtr = std::shared_ptr<const KFState>;

  KFState();

  KFState(const KFState &b);

  KFState &operator=(const KFState &b) = default;

  KFState operator+(const VecDIM &state_add);

  KFState &operator+=(const VecDIM &state_add);

  VecDIM operator-(const KFState &b);

  void resetPVQ();

  double timestamp;

  Vec3 sw_pos_[WINDOW_SIZE];  // the estimated position of IMU at World frame of history state, unit: m
  Mat3 sw_rot_[WINDOW_SIZE];  // the estimated attitude of IMU at World frame of history state, unit: rad
  Vec3 vel_;                  // the estimated velocity of IMU at World frame, unit: m/s^2
  Vec3 gyo_bias_;             // gyroscope bias, unit: rad/s
  Vec3 acc_bias_;             // accelerator bias, unit: m/s^2
  double sw_timestamp[WINDOW_SIZE];

#if GRAVITY_CALIBRATION
  Vec3 gravity;  // gravity
#endif

#if EXTRINSIC_CALIBRATION
  Mat3 R_I_L;       // the estimated attitude (rotation matrix) of extrinsic
  Vec3 t_I_L;       // the estimated position of extrinsic
#endif
  MatDIM mill_cov;  // states covariance
};
}  // namespace lixel
