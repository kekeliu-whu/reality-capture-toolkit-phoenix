#include <spdlog/spdlog.h>

#include <algorithm>
#include <numeric>

#include "imu_processing.h"

namespace {

using AlignedV3DVector = std::vector<V3D, Eigen::aligned_allocator<V3D>>;

void ComputeMeanAndStd(const AlignedV3DVector &samples, const std::vector<size_t> &indices,
                       V3D &mean, V3D &stddev) {
  mean.setZero();
  stddev.setZero();
  if (indices.empty()) return;

  for (const size_t index : indices) mean += samples[index];
  mean /= static_cast<double>(indices.size());

  if (indices.size() < 2) return;
  for (const size_t index : indices) {
    const V3D residual = samples[index] - mean;
    stddev += residual.cwiseProduct(residual);
  }
  stddev = (stddev / static_cast<double>(indices.size() - 1)).cwiseSqrt();
}

}  // namespace

ImuProcess::ImuProcess() : b_first_frame_(true), imu_need_init_(true), start_timestamp_(-1) {
  Q               = process_noise_cov();
  cov_acc         = V3D(0.1, 0.1, 0.1);
  cov_gyr         = V3D(0.1, 0.1, 0.1);
  cov_bias_gyr    = V3D(0.0001, 0.0001, 0.0001);
  cov_bias_acc    = V3D(0.0001, 0.0001, 0.0001);
  mean_acc        = V3D(0, 0, -1.0);
  mean_gyr        = V3D(0, 0, 0);
  angvel_last     = V3D::Zero();
  Lidar_T_wrt_IMU = V3D::Zero();
  Lidar_R_wrt_IMU = M3D::Identity();
  last_imu_.reset(new sensor_msgs::Imu());
  last_lidar_end_time_ = 0;
}

ImuProcess::~ImuProcess() {}

void ImuProcess::Reset() {
  mean_acc         = V3D(0, 0, -1.0);
  mean_gyr         = V3D(0, 0, 0);
  angvel_last      = V3D::Zero();
  imu_need_init_   = true;
  start_timestamp_ = -1;
  last_lidar_end_time_ = 0;
  init_acc_samples_.clear();
  init_gyr_samples_.clear();
  init_wait_logged_ = false;
  v_imu_.clear();
  IMUpose.clear();
  last_imu_.reset(new sensor_msgs::Imu());
}

void ImuProcess::set_extrinsic(const MD(4, 4) & T) {
  Lidar_T_wrt_IMU = T.block<3, 1>(0, 3);
  Lidar_R_wrt_IMU = T.block<3, 3>(0, 0);
}

void ImuProcess::set_extrinsic(const V3D &transl) {
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU.setIdentity();
}

void ImuProcess::set_extrinsic(const V3D &transl, const M3D &rot) {
  Lidar_T_wrt_IMU = transl;
  Lidar_R_wrt_IMU = rot;
}

void ImuProcess::set_gyr_cov(const V3D &scaler) { cov_gyr_scale = scaler; }

void ImuProcess::set_acc_cov(const V3D &scaler) { cov_acc_scale = scaler; }

void ImuProcess::set_gyr_bias_cov(const V3D &b_g) { cov_bias_gyr = b_g; }

void ImuProcess::set_acc_bias_cov(const V3D &b_a) { cov_bias_acc = b_a; }

void ImuProcess::set_dynamic_initialization_options(const DynamicInitializationOptions &options) {
  dynamic_init_options_ = options;
}

bool ImuProcess::TryInitialize(const MeasureGroup &meas,
                               esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state) {
  if (b_first_frame_) {
    Reset();
    b_first_frame_   = false;
    first_lidar_time = meas.lidar_beg_time;
  }

  for (const auto &imu : meas.imu) {
    const auto &imu_acc = imu->linear_acceleration;
    const auto &gyr_acc = imu->angular_velocity;
    V3D cur_acc(imu_acc.x, imu_acc.y, imu_acc.z);
    V3D cur_gyr(gyr_acc.x, gyr_acc.y, gyr_acc.z);
    if (!cur_acc.allFinite() || !cur_gyr.allFinite()) continue;
    if (start_timestamp_ < 0) start_timestamp_ = imu->header.stamp.toSec();
    init_acc_samples_.push_back(cur_acc);
    init_gyr_samples_.push_back(cur_gyr);
  }

  if (init_acc_samples_.size() < 20 || start_timestamp_ < 0) return false;

  const double end_timestamp = meas.imu.back()->header.stamp.toSec();
  const double duration_sec  = end_timestamp - start_timestamp_;
  if (dynamic_init_options_.enabled && duration_sec < dynamic_init_options_.min_duration_sec) return false;

  std::vector<size_t> all_indices(init_acc_samples_.size());
  std::iota(all_indices.begin(), all_indices.end(), 0);

  V3D acc_all_mean, acc_all_std, gyr_all_mean, gyr_all_std;
  ComputeMeanAndStd(init_acc_samples_, all_indices, acc_all_mean, acc_all_std);
  ComputeMeanAndStd(init_gyr_samples_, all_indices, gyr_all_mean, gyr_all_std);

  const bool stationary =
      !dynamic_init_options_.enabled ||
      (acc_all_std.norm() <= dynamic_init_options_.max_acc_std_mps2 &&
       gyr_all_std.norm() <= dynamic_init_options_.max_gyr_std_radps &&
       gyr_all_mean.norm() <= dynamic_init_options_.max_mean_gyr_radps &&
       std::abs(acc_all_mean.norm() - G_m_s2) <= dynamic_init_options_.max_acc_norm_error_mps2);

  if (stationary) {
    FinishInitialization(meas, kf_state, acc_all_mean, gyr_all_mean, acc_all_std, gyr_all_std, true);
    return true;
  }

  if (duration_sec < dynamic_init_options_.max_duration_sec) {
    if (!init_wait_logged_) {
      spdlog::info(
          "[INIT] Motion detected after {:.2f}s; collecting IMU until {:.2f}s before dynamic initialization "
          "(acc_std={:.4f}, gyr_std={:.4f}, mean_gyr={:.4f}).",
          duration_sec, dynamic_init_options_.max_duration_sec, acc_all_std.norm(), gyr_all_std.norm(),
          gyr_all_mean.norm());
      init_wait_logged_ = true;
    }
    return false;
  }

  // Gravity and IMU biases are not jointly observable from a moving IMU alone.
  // Select the least dynamic samples to obtain a robust roll/pitch seed, but do
  // not interpret the mean angular velocity as gyro bias in this mode.
  std::vector<size_t> robust_indices = all_indices;
  std::sort(robust_indices.begin(), robust_indices.end(), [&](const size_t lhs, const size_t rhs) {
    const double lhs_score = std::abs(init_acc_samples_[lhs].norm() - G_m_s2) /
                                 std::max(0.1, dynamic_init_options_.max_acc_norm_error_mps2) +
                             init_gyr_samples_[lhs].norm() /
                                 std::max(0.01, dynamic_init_options_.max_mean_gyr_radps);
    const double rhs_score = std::abs(init_acc_samples_[rhs].norm() - G_m_s2) /
                                 std::max(0.1, dynamic_init_options_.max_acc_norm_error_mps2) +
                             init_gyr_samples_[rhs].norm() /
                                 std::max(0.01, dynamic_init_options_.max_mean_gyr_radps);
    return lhs_score < rhs_score;
  });
  const size_t keep_count = std::min(
      robust_indices.size(),
      std::max<size_t>(20, static_cast<size_t>(std::ceil(
                               robust_indices.size() * dynamic_init_options_.robust_sample_ratio))));
  robust_indices.resize(keep_count);

  V3D acc_robust_mean, acc_robust_std, gyr_robust_mean, gyr_robust_std;
  ComputeMeanAndStd(init_acc_samples_, robust_indices, acc_robust_mean, acc_robust_std);
  ComputeMeanAndStd(init_gyr_samples_, robust_indices, gyr_robust_mean, gyr_robust_std);
  FinishInitialization(meas, kf_state, acc_robust_mean, V3D::Zero(), acc_robust_std, gyr_robust_std, false);
  return true;
}

void ImuProcess::FinishInitialization(const MeasureGroup &meas,
                                      esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                                      const V3D &acc_reference, const V3D &gyr_reference,
                                      const V3D &acc_std, const V3D &gyr_std, bool stationary) {
  mean_acc = acc_reference;
  mean_gyr = gyr_reference;
  cov_acc  = acc_std.cwiseProduct(acc_std);
  cov_gyr  = gyr_std.cwiseProduct(gyr_std);

  V3D specific_force_direction = mean_acc;
  if (specific_force_direction.norm() < 1e-6) specific_force_direction = V3D(0, 0, G_m_s2);
  specific_force_direction.normalize();
  const V3D gravity_specific_force = specific_force_direction * G_m_s2;

  state_ikfom init_state = kf_state.get_x();
  init_state.grav        = S2(V3D(0, 0, -G_m_s2));
  init_state.rot         = Eigen::Quaterniond::FromTwoVectors(-gravity_specific_force, V3D(0, 0, -1.0));
  init_state.vel         = V3D::Zero();
  if (stationary) {
    init_state.ba = mean_acc - gravity_specific_force;
    init_state.bg = mean_gyr;
  } else {
    init_state.ba.setZero();
    init_state.bg.setZero();
  }
  init_state.offset_T_L_I = Lidar_T_wrt_IMU;
  init_state.offset_R_L_I = Lidar_R_wrt_IMU;
  kf_state.change_x(init_state);

  esekfom::esekf<state_ikfom, 12, input_ikfom>::cov init_P = kf_state.get_P();
  init_P.setIdentity();
  init_P(6, 6) = init_P(7, 7) = init_P(8, 8) = 0.00001;
  init_P(9, 9) = init_P(10, 10) = init_P(11, 11) = 0.00001;
  init_P(15, 15) = init_P(16, 16) = init_P(17, 17) = stationary ? 0.0001 : 0.01;
  init_P(18, 18) = init_P(19, 19) = init_P(20, 20) = stationary ? 0.001 : 0.10;
  init_P(21, 21) = init_P(22, 22) = stationary ? 0.00001 : 0.01;
  kf_state.change_P(init_P);

  const auto &last_acc_msg = meas.imu.back()->linear_acceleration;
  const auto &last_gyr_msg = meas.imu.back()->angular_velocity;
  const V3D last_acc(last_acc_msg.x, last_acc_msg.y, last_acc_msg.z);
  const V3D last_gyr(last_gyr_msg.x, last_gyr_msg.y, last_gyr_msg.z);
  angvel_last = last_gyr - init_state.bg;
  acc_s_last  = init_state.rot * (last_acc - init_state.ba) + V3D(0, 0, -G_m_s2);
  last_imu_   = meas.imu.back();
  last_lidar_end_time_ = meas.lidar_end_time;
  imu_need_init_ = false;

  const double duration_sec = meas.imu.back()->header.stamp.toSec() - start_timestamp_;
  spdlog::info(
      "[INIT] Dynamic initialization complete: mode={}, duration={:.3f}s, samples={}, "
      "acc=[{:.6f}, {:.6f}, {:.6f}], acc_std={:.6f}, gyr_std={:.6f}, "
      "ba=[{:.6f}, {:.6f}, {:.6f}], bg=[{:.6f}, {:.6f}, {:.6f}].",
      stationary ? "stationary" : "moving", duration_sec, init_acc_samples_.size(), mean_acc.x(), mean_acc.y(),
      mean_acc.z(), acc_std.norm(), gyr_std.norm(), init_state.ba.x(), init_state.ba.y(), init_state.ba.z(),
      init_state.bg.x(), init_state.bg.y(), init_state.bg.z());

  cov_acc = cov_acc_scale;
  cov_gyr = cov_gyr_scale;
}

void ImuProcess::UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                              PointCloudXYZI &pcl_out) {
  /*** add the imu of the last frame-tail to the of current frame-head ***/
  auto v_imu = meas.imu;
  v_imu.push_front(last_imu_);
  const double &imu_beg_time = v_imu.front()->header.stamp.toSec();
  const double &imu_end_time = v_imu.back()->header.stamp.toSec();
  const double &pcl_beg_time = meas.lidar_beg_time;
  const double &pcl_end_time = meas.lidar_end_time;

  /*** sort point clouds by offset time ***/
  pcl_out = *(meas.lidar);
  sort(pcl_out.points.begin(), pcl_out.points.end(), time_list);
  // spdlog::info("[ IMU Process ]: Process lidar from {} to {}, {} imu msgs from {} to {}", pcl_beg_time, pcl_end_time,
  // meas.imu.size(), imu_beg_time, imu_end_time);

  /*** Initialize IMU pose ***/
  state_ikfom imu_state = kf_state.get_x();
  IMUpose.clear();
  IMUpose.push_back(
      set_pose6d(0.0, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));

  /*** forward propagation at each imu point ***/
  V3D angvel_avr, acc_avr, acc_imu, vel_imu, pos_imu;
  M3D R_imu;

  double dt = 0;

  input_ikfom in;
  for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
    auto &&head = *(it_imu);
    auto &&tail = *(it_imu + 1);

    if (tail->header.stamp.toSec() < last_lidar_end_time_) continue;

    angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x),
        0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
        0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
    acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x),
        0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
        0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

    if (head->header.stamp.toSec() < last_lidar_end_time_) {
      dt = tail->header.stamp.toSec() - last_lidar_end_time_;
    } else {
      dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
    }

    in.acc                         = acc_avr;
    in.gyro                        = angvel_avr;
    Q.block<3, 3>(0, 0).diagonal() = cov_gyr;
    Q.block<3, 3>(3, 3).diagonal() = cov_acc;
    Q.block<3, 3>(6, 6).diagonal() = cov_bias_gyr;
    Q.block<3, 3>(9, 9).diagonal() = cov_bias_acc;
    kf_state.predict(dt, Q, in);

    /* save the poses at each IMU measurements */
    imu_state   = kf_state.get_x();
    angvel_last = angvel_avr - imu_state.bg;
    acc_s_last  = imu_state.rot * (acc_avr - imu_state.ba);
    for (int i = 0; i < 3; i++) {
      acc_s_last[i] += imu_state.grav[i];
    }
    double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
    IMUpose.push_back(
        set_pose6d(offs_t, acc_s_last, angvel_last, imu_state.vel, imu_state.pos, imu_state.rot.toRotationMatrix()));
  }

  /*** calculated the pos and attitude prediction at the frame-end ***/
  double note = pcl_end_time > imu_end_time ? 1.0 : -1.0;
  dt          = note * (pcl_end_time - imu_end_time);
  kf_state.predict(dt, Q, in);

  imu_state            = kf_state.get_x();
  last_imu_            = meas.imu.back();
  last_lidar_end_time_ = pcl_end_time;

  /*** undistort each lidar point (backward propagation) ***/
  if (pcl_out.points.begin() == pcl_out.points.end()) return;
  auto it_pcl = pcl_out.points.end() - 1;
  for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
    auto head = it_kp - 1;
    auto tail = it_kp;
    R_imu = head->rot;
    // spdlog::info("head imu acc: {}", acc_imu.transpose());
    vel_imu << VEC_FROM_ARRAY(head->vel);
    pos_imu << VEC_FROM_ARRAY(head->pos);
    acc_imu << VEC_FROM_ARRAY(tail->acc);
    angvel_avr << VEC_FROM_ARRAY(tail->gyr);

    for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
      dt = it_pcl->curvature / double(1000) - head->offset_time;

      /* Transform to the 'end' frame, using only the rotation
       * Note: Compensation direction is INVERSE of Frame's moving direction
       * So if we want to compensate a point at timestamp-i to the frame-e
       * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is represented in global frame */
      M3D R_i(R_imu * Exp(angvel_avr, dt));

      V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
      V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt - imu_state.pos);
      V3D P_compensate =
          imu_state.offset_R_L_I.conjugate() *
          (imu_state.rot.conjugate() * (R_i * (imu_state.offset_R_L_I * P_i + imu_state.offset_T_L_I) + T_ei) -
           imu_state.offset_T_L_I);  // not accurate!

      // save Undistorted points and their rotation
      it_pcl->x = P_compensate(0);
      it_pcl->y = P_compensate(1);
      it_pcl->z = P_compensate(2);

      if (it_pcl == pcl_out.points.begin()) break;
    }
  }
}

bool ImuProcess::Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                         PointCloudXYZI::Ptr cur_pcl_un) {
  double t1, t2, t3;
  t1 = omp_get_wtime();

  if (cur_pcl_un) cur_pcl_un->clear();

  if (meas.imu.empty()) {
    return false;
  };
  if (meas.lidar == nullptr) {
    spdlog::error("meas.lidar == nullptr");
    exit(1);
  }

  if (imu_need_init_) {
    TryInitialize(meas, kf_state);
    return false;
  }

  UndistortPcl(meas, kf_state, *cur_pcl_un);

  t2 = omp_get_wtime();
  t3 = omp_get_wtime();

  // spdlog::info("[ IMU Process ]: Time: {}", t3 - t1);
  return !cur_pcl_un->empty();
}
