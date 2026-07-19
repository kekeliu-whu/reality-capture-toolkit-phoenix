#include "ieskf/ieskf.h"

namespace lixel
{

void IESKF::update(BaseFusion& fusion, AttributeIterate& attr_iter)
{
  if (!init_)
  {
    lslog(LSLOG_ERROR) << "IESKF not init!";
    return;
  }

  bool flg_EKF_converged = false;

  KFState state_propagat = *states_ptr_;
  VecDIM dx_last_iter;
  VecDIM dx_update;
  float std_last_iter;
  int iter_counter = 0;
  // LOG(INFO) << "before_P:\n" << states_ptr_->mill_cov;
  for (iter_counter = 0; iter_counter < ieskf_configs_.max_iter_num; ++iter_counter)
  {
    TicToc init_ts;
    double mill_R, R;
    VecX residual;
    SparseMat H;
    MatDIM& mill_P = states_ptr_->mill_cov;
    fusion.setCurState(*states_ptr_);
    init_ts_ += init_ts.Toc();
    TicToc registration_ts;
    fusion.calculateMeas(residual, H, R);
    mill_R = R * SCALE;
    if (residual.rows() == 0)
    {
      lslog(LSLOG_ERROR) << "effect points equals to 0, stop current update";
      return;
    }

    registration_ts_ += registration_ts.Toc();

    TicToc matrix_calculation_time_cost;
    if (residual.rows() != H.rows())
    {
      lslog(LSLOG_ERROR) << "Matrix dimensions of H, R, reisudal not satisfied";
      return;
    }
    /** Sparse Matrix Calculation about K = (H^T * R^-1 * H + P^-1)-1 H^T * R^-1 **/
    SparseMat HT_Rinv = H.transpose() / mill_R;
    MatDIMX K = ((MatDIM)(HT_Rinv * H) + mill_P.inverse()).inverse() * HT_Rinv;
    SparseMat K_spare = K.sparseView();
    // LOG(INFO) << "K:\n" << K;
    // LOG(INFO) << "H_sparse:\n" << H_sparse;

    /*** x^k+1 = x^k + ((-Kz) - (I-KH)(x^k - x)) ***/
    VecDIM deltax = -K_spare * residual - (MatDIM::Identity() - K_spare * H) * (*states_ptr_ - state_propagat);
    (*states_ptr_) += deltax;

#if GRAVITY_CALIBRATION
    states_ptr_->gravity.normalize();
    states_ptr_->gravity *= DEFAULT_GRAVITY;
#endif
    /*** Converged Judgement ***/
    Vec3 rot_increment = deltax.block<3, 1>(0, 0);
    Vec3 t_increment = deltax.block<3, 1>(3, 0);
    if ((rot_increment.norm() < CONVERAGE_ROT_THRESHOLD) && (t_increment.norm() < CONVERAGE_TRANS_THRESHOLD))
      flg_EKF_converged = true;

    if (flg_EKF_converged || (iter_counter == ieskf_configs_.max_iter_num - 1))
    {
      /*** Covariance Update ***/
      mill_P = (MatDIM::Identity() - K * H) * mill_P;
      lslog(LSLOG_INFO) << "kf converged:" << flg_EKF_converged;
      dx_last_iter = deltax;
      std_last_iter = sqrt(mill_R / SCALE);
      break;
    }
    matrix_calculation_ts_ += matrix_calculation_time_cost.Toc();
  }
  /*** LOG intermediate results of IESKF ***/
  dx_update = *states_ptr_ - state_propagat;
  attr_iter.iter_num = iter_counter;
  attr_iter.rot_tol = dx_last_iter.block<3, 1>(0, 0).cast<float>();
  attr_iter.pos_tol = dx_last_iter.block<3, 1>(3, 0).cast<float>();
  attr_iter.std_dev = std_last_iter;

  attr_iter.rot_update = dx_update.block<3, 1>(0, 0);
  attr_iter.pos_update = dx_update.block<3, 1>(3, 0);
  attr_iter.vel_update = dx_update.block<3, 1>(6, 0);
  attr_iter.gyro_bias_update = dx_update.block<3, 1>(9, 0);
  attr_iter.acc_bias_update = dx_update.block<3, 1>(12, 0);

  VecDIM std_diag = states_ptr_->mill_cov.diagonal().array().sqrt();
  attr_iter.rot_std = std_diag.block<3, 1>(DIM_STATE - 6, 0);
  attr_iter.pos_std = std_diag.block<3, 1>(DIM_STATE - 3, 0);
  attr_iter.vel_std = std_diag.block<3, 1>(6, 0);
  attr_iter.gyro_bias_std = std_diag.block<3, 1>(9, 0);
  attr_iter.acc_bias_std = std_diag.block<3, 1>(12, 0);
  attr_iter.gravity_std = V3D::Zero();

  lslog(LSLOG_INFO) << "registrationT:" << registration_ts_;
  lslog(LSLOG_INFO) << "matrixT:" << matrix_calculation_ts_;
  registration_ts_ = 0;
  init_ts_ = 0;
  matrix_calculation_ts_ = 0;
  // logState("update_");
}

}  // namespace lixel
