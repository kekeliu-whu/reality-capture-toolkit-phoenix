//
// Created by youyuan on 24-2-4.
//

#pragma once

#define GRAVITY_CALIBRATION false
#define EXTRINSIC_CALIBRATION false

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
constexpr int DIM_CURR_STATE = getDIMCurrState();
constexpr double INIT_COV = 1e-4;

using VecDIM = Eigen::Matrix<FloatDataType, Eigen::Dynamic, 1>;
using MatDIM = Eigen::Matrix<FloatDataType, Eigen::Dynamic, Eigen::Dynamic>;
using VecX = Eigen::Matrix<FloatDataType, Eigen::Dynamic, 1>;
using MatXDIM = Eigen::Matrix<FloatDataType, Eigen::Dynamic, Eigen::Dynamic>;
using MatDIMX = Eigen::Matrix<FloatDataType, Eigen::Dynamic, Eigen::Dynamic>;
using SparseMat = Eigen::SparseMatrix<FloatDataType, Eigen::RowMajor>;
// clang-format on

/*** State Order: sw_rot***/
class KFState
{
 public:
  using Ptr = std::shared_ptr<KFState>;
  using ConstPtr = std::shared_ptr<const KFState>;

  explicit KFState(int window_size = 10);

  KFState(const KFState &b) = default;

  KFState &operator=(const KFState &b) = default;

  KFState operator+(const VecDIM &state_add);

  KFState &operator+=(const VecDIM &state_add);

  VecDIM operator-(const KFState &b);

  void resetPVQ();
  int windowSize() const;
  int dimState() const;

  double timestamp;

  std::vector<Vec3, Eigen::aligned_allocator<Vec3>> sw_pos_;  // IMU positions in the world frame
  std::vector<Mat3, Eigen::aligned_allocator<Mat3>> sw_rot_;  // IMU attitudes in the world frame
  Vec3 vel_;                  // the estimated velocity of IMU at World frame, unit: m/s^2
  Vec3 gyo_bias_;             // gyroscope bias, unit: rad/s
  Vec3 acc_bias_;             // accelerator bias, unit: m/s^2
  std::vector<double> sw_timestamp;

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
