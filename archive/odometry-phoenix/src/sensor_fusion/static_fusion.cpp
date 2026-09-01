//
// Created by youyuan on 24-2-18.
//

#include "sensor_fusion/static_fusion.h"
#include "xmap_util.h"

namespace lixel
{

void StaticFusion::calculateMeas(VecX& residual, SparseMat& H, double& R)
{
  R = STATIC_COV;
  residual = Eigen::Matrix<FloatDataType, STATIC_MEASUREMENT_DIM, 1>::Zero();

  Eigen::Matrix<FloatDataType, 9, Eigen::Dynamic> H_Sparse =
      Eigen::Matrix<FloatDataType, 9, Eigen::Dynamic>::Zero(
          STATIC_MEASUREMENT_DIM, states_group_ptr_->dimState());

  // static means PV not change
  residual.block<3, 1>(3, 0) = init_t_ - states_group_ptr_->sw_pos_[0];
  residual.block<3, 1>(6, 0) = Vec3::Zero() - states_group_ptr_->vel_;
  H_Sparse.block<3, 3>(3, 3) = -Mat3::Identity();
  H_Sparse.block<3, 3>(6, 6) = -Mat3::Identity();
  H = H_Sparse.sparseView();
#if UPDATE_ROT
  residual.block<3, 1>(0, 0) = log(init_R_ * states_group_ptr_->rot.transpose());
  H.block<3, 3>(0, 0) = Mat3::Identity();
#endif
}

void StaticFusion::setInitRT(const Mat3& R, const Vec3& t)
{
  init_R_ = R;
  init_t_ = t;
}

}  // namespace lixel
