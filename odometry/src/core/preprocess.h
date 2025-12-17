#pragma once

#include <spdlog/spdlog.h>
#include <optional>
#include <sophus/se3.hpp>

#include "common/types.h"

#define G_m_s2 (9.81)  // Gravaty const in GuangDong/China
#define MD(a, b) Eigen::Matrix<double, (a), (b)>
#define VD(a) Eigen::Matrix<double, (a), 1>

void ProcessRawSensorData(const SensorCalib &calib, MsgPack &msg_pack);

class ImuPreprocess {
 public:
  bool TryInit(const std::vector<ImuMsg> &imu_msgs, StatesGroup &stat) {
    if (!first_timestamp_) {
      first_timestamp_ = imu_msgs.front().timestamp;
    }

    if (imu_msgs.back().timestamp - *first_timestamp_ < kInitDuration) {
      IMU_init(imu_msgs, stat, init_iter_num);

      spdlog::info("IMU Init: iter num: {}, mean acc: {}, mean gyr: {}", init_iter_num, mean_acc.transpose(), mean_gyr.transpose());

      return false;
    }

    return true;
  }

  void IMU_init(const std::vector<ImuMsg> &imu_msgs, StatesGroup &state_inout, int &N) {
    for (const auto &imu : imu_msgs) {
      N++;

      const auto &cur_acc = imu.acc;
      const auto &cur_gyr = imu.gyr;

      mean_acc += (cur_acc - mean_acc) / N;
      mean_gyr += (cur_gyr - mean_gyr) / N;
    }

    state_inout.gravity = -mean_acc / mean_acc.norm() * G_m_s2;

    state_inout.rot_end = Matrix3::Identity();  // Exp(mean_acc.cross(Vector3(0, 0, -1 / scale_gravity)));
    state_inout.bias_g  = mean_gyr;
  }

  struct Pose6D {
    double timestamp;
    Sophus::SE3d pose;

    Pose6D(double timestamp, Quaternion rot, Vector3 pos) : timestamp(timestamp), pose(Sophus::SE3d(rot, pos)) {}
  };

  void Process(const MsgPack &msg_pack, StatesGroup &state_inout, PointCloud::Ptr &pcl_out) {
    auto imu_vec = GetInterpolatedImuVec(msg_pack);

    /*** add the imu of the last frame-tail to the of current frame-head ***/
    *pcl_out = *msg_pack.lidar_points;

    /*** Initialize IMU pose ***/
    std::vector<Pose6D> IMUpose;
    IMUpose.push_back(Pose6D(imu_vec.front().timestamp, Quaternion(state_inout.rot_end), state_inout.pos_end));

    /*** forward propagation at each imu point ***/
    Vector3 acc_imu, angvel_avr, acc_avr, vel_imu(state_inout.vel_end), pos_imu(state_inout.pos_end);
    Matrix3 R_imu(state_inout.rot_end);
    MD(DIM_STATE, DIM_STATE) F_x, cov_w;

    for (auto it_imu = v_imu.begin(); it_imu < (v_imu.end() - 1); it_imu++) {
      auto &&head = *(it_imu);
      auto &&tail = *(it_imu + 1);

      if (tail->header.stamp.toSec() < pcl_beg_time) continue;

      angvel_avr << 0.5 * (head->angular_velocity.x + tail->angular_velocity.x), 0.5 * (head->angular_velocity.y + tail->angular_velocity.y),
          0.5 * (head->angular_velocity.z + tail->angular_velocity.z);
      acc_avr << 0.5 * (head->linear_acceleration.x + tail->linear_acceleration.x), 0.5 * (head->linear_acceleration.y + tail->linear_acceleration.y),
          0.5 * (head->linear_acceleration.z + tail->linear_acceleration.z);

      angvel_avr -= state_inout.bias_g;
      acc_avr = acc_avr * G_m_s2 / mean_acc.norm() - state_inout.bias_a;

      if (head->header.stamp.toSec() < pcl_beg_time) {
        dt = tail->header.stamp.toSec() - pcl_beg_time;
      } else {
        dt = tail->header.stamp.toSec() - head->header.stamp.toSec();
      }

      /* covariance propagation */
      Matrix3 acc_avr_skew;
      Matrix3 Exp_f = Exp(angvel_avr, dt);
      acc_avr_skew << SKEW_SYM_MATRX(acc_avr);

      F_x.setIdentity();
      cov_w.setZero();

      F_x.block<3, 3>(0, 0) = Exp(angvel_avr, -dt);
      F_x.block<3, 3>(0, 9) = -Matrix3::Identity() * dt;
      // F_x.block<3,3>(3,0)  = R_imu * off_vel_skew * dt;
      F_x.block<3, 3>(3, 6)  = Matrix3::Identity() * dt;
      F_x.block<3, 3>(6, 0)  = -R_imu * acc_avr_skew * dt;
      F_x.block<3, 3>(6, 12) = -R_imu * dt;
      F_x.block<3, 3>(6, 15) = Matrix3::Identity() * dt;

      cov_w.block<3, 3>(0, 0).diagonal()   = cov_gyr * dt * dt;
      cov_w.block<3, 3>(6, 6)              = R_imu * cov_acc.asDiagonal() * R_imu.transpose() * dt * dt;
      cov_w.block<3, 3>(9, 9).diagonal()   = cov_bias_gyr * dt * dt;  // bias gyro covariance
      cov_w.block<3, 3>(12, 12).diagonal() = cov_bias_acc * dt * dt;  // bias acc covariance

      state_inout.cov = F_x * state_inout.cov * F_x.transpose() + cov_w;

      /* propogation of IMU attitude */
      R_imu = R_imu * Exp_f;

      /* Specific acceleration (global frame) of IMU */
      acc_imu = R_imu * acc_avr + state_inout.gravity;

      /* propogation of IMU */
      pos_imu = pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt;

      /* velocity of IMU */
      vel_imu = vel_imu + acc_imu * dt;

      /* save the poses at each IMU measurements */
      angvel_last     = angvel_avr;
      acc_s_last      = acc_imu;
      double &&offs_t = tail->header.stamp.toSec() - pcl_beg_time;
      IMUpose.push_back(set_pose6d(offs_t, acc_imu, angvel_avr, vel_imu, pos_imu, R_imu));
    }

    /*** calculated the pos and attitude prediction at the frame-end ***/
    double note         = pcl_end_time > imu_end_time ? 1.0 : -1.0;
    dt                  = note * (pcl_end_time - imu_end_time);
    state_inout.vel_end = vel_imu + note * acc_imu * dt;
    state_inout.rot_end = R_imu * Exp(V3D(note * angvel_avr), dt);
    state_inout.pos_end = pos_imu + note * vel_imu * dt + note * 0.5 * acc_imu * dt * dt;

    auto pos_liD_e = state_inout.pos_end + state_inout.rot_end * Lid_offset_to_IMU;

    /*** undistort each lidar point (backward propagation) ***/
    auto it_pcl = pcl_out.points.end() - 1;
    for (auto it_kp = IMUpose.end() - 1; it_kp != IMUpose.begin(); it_kp--) {
      auto head = it_kp - 1;
      auto tail = it_kp;
      R_imu = head->rot;
      acc_imu << VEC_FROM_ARRAY(head->acc);
      // cout<<"head imu acc: "<<acc_imu.transpose()<<endl;
      vel_imu << VEC_FROM_ARRAY(head->vel);
      pos_imu << VEC_FROM_ARRAY(head->pos);
      angvel_avr << VEC_FROM_ARRAY(head->gyr);

      for (; it_pcl->curvature / double(1000) > head->offset_time; it_pcl--) {
        dt = it_pcl->curvature / double(1000) - head->offset_time;

        /* Transform to the 'end' frame, using only the rotation
         * Note: Compensation direction is INVERSE of Frame's moving direction
         * So if we want to compensate a point at timestamp-i to the frame-e
         * P_compensate = R_imu_e ^ T * (R_i * P_i + T_ei) where T_ei is
         * represented in global frame */
        M3D R_i(R_imu * Exp(angvel_avr, dt));
        V3D T_ei(pos_imu + vel_imu * dt + 0.5 * acc_imu * dt * dt + R_i * Lid_offset_to_IMU - pos_liD_e);

        V3D P_i(it_pcl->x, it_pcl->y, it_pcl->z);
        V3D P_compensate = state_inout.rot_end.transpose() * (R_i * P_i + T_ei);

        /// save Undistorted points and their rotation
        it_pcl->x = P_compensate(0);
        it_pcl->y = P_compensate(1);
        it_pcl->z = P_compensate(2);

        if (it_pcl == pcl_out.points.begin()) break;
      }
    }
  }

 private:
  std::vector<ImuMsg> GetInterpolatedImuVec(const MsgPack &msg_pack) {
    std::vector<ImuMsg> imu_vec;

    const auto &imu_before_start = msg_pack.imu_msgs[0];
    const auto &imu_after_start  = msg_pack.imu_msgs[1];

    double alpha_start = (msg_pack.group_start_time - imu_before_start.timestamp) / (imu_after_start.timestamp - imu_before_start.timestamp);

    ImuMsg start_imu;
    start_imu.timestamp = msg_pack.group_start_time;
    start_imu.acc       = imu_before_start.acc + alpha_start * (imu_after_start.acc - imu_before_start.acc);
    start_imu.gyr       = imu_before_start.gyr + alpha_start * (imu_after_start.gyr - imu_before_start.gyr);

    imu_vec.push_back(start_imu);

    for (size_t i = 1; i < msg_pack.imu_msgs.size() - 1; ++i) {
      if (msg_pack.imu_msgs[i].timestamp > msg_pack.group_start_time && msg_pack.imu_msgs[i].timestamp < msg_pack.group_end_time) {
        imu_vec.push_back(msg_pack.imu_msgs[i]);
      }
    }

    const auto &imu_before_end = msg_pack.imu_msgs[msg_pack.imu_msgs.size() - 2];
    const auto &imu_after_end  = msg_pack.imu_msgs[msg_pack.imu_msgs.size() - 1];

    double alpha_end = (msg_pack.group_end_time - imu_before_end.timestamp) / (imu_after_end.timestamp - imu_before_end.timestamp);

    ImuMsg end_imu;
    end_imu.timestamp = msg_pack.group_end_time;
    end_imu.acc       = imu_before_end.acc + alpha_end * (imu_after_end.acc - imu_before_end.acc);
    end_imu.gyr       = imu_before_end.gyr + alpha_end * (imu_after_end.gyr - imu_before_end.gyr);

    imu_vec.push_back(end_imu);

    return imu_vec;
  }

 private:
  std::optional<double> first_timestamp_;
  // todo kk init cov_acc
  Vector3 cov_acc;
  Vector3 cov_gyr;
  Vector3 cov_bias_gyr;
  Vector3 cov_bias_acc;
  Vector3 mean_acc  = Vector3::Zero();
  Vector3 mean_gyr  = Vector3::Zero();
  int init_iter_num = 0;

  static constexpr int kInitDuration = 2;  // seconds
};
