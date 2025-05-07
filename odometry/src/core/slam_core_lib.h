#pragma once

#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>

#include "common/types.h"
#include "core/preprocess.h"
#include "io/msg_pack_synchronizer.h"

class SlamCore {
 public:
  SlamCore(const SensorCalib &calib) : calib_(calib), msg_pack_synchronizer_(new MsgPackSynchronizer(calib.has_encoder)) {}

  void AddSensorData(LidarMsg::ConstPtr lidar_msg) { msg_pack_synchronizer_->AddLidarData(lidar_msg); }

  void AddSensorData(ImuMsg imu_msg) { msg_pack_synchronizer_->AddImuData(imu_msg); }

  void AddSensorData(EncoderMsg encoder_msg) { msg_pack_synchronizer_->AddEncoderData(encoder_msg); }

  void TryEstimateState(OdometryResult::Ptr &result) {
    result.reset(new OdometryResult);

    MsgPack msg_pack;
    if (!msg_pack_synchronizer_->SyncMsgPack(msg_pack)) {
      return;
    }

    ProcessRawSensorData(calib_, msg_pack);

    pcl::io::savePCDFileBinary("/debug/" + std::to_string(msg_pack.id) + ".pcd", *msg_pack.lidar_points);

    // 1. try init map and imu using first N seconds data
    if (!p_imu->TryInit(msg_pack.imu_msgs, state)) {
      // incremental init map
      return;
    }

    // // 2. propagate state using IMU data
    // PointCloud::Ptr feats_body_undistort(new PointCloud);
    // p_imu->Process(msg_pack, state, feats_body_undistort);

    // // 3. downsample scan size
    // PointCloud::Ptr feats_undistort_body_down;
    // PointCloud::Ptr feats_undistort_body = msg_pack.lidar_points;

    // 4. build match pairs

    // int rematch_num        = 0;
    // bool nearest_search_en = true;
    // state_propagat = state;
    // int NUM_MAX_ITERATIONS = 6;
    // for (iterCount = 0; iterCount < NUM_MAX_ITERATIONS; iterCount++) {
    //   laserCloudOri->clear();
    //   laserCloudNoeffect->clear();
    //   corr_normvect->clear();

    //   std::vector<double> r_list;
    //   std::vector<ptpl> ptpl_list;
    //   /** LiDAR match based on 3 sigma criterion **/

    //   vector<pointWithCov> pv_list;
    //   std::vector<Matrix3> var_list;
    //   pcl::PointCloud<pcl::PointXYZI>::Ptr world_lidar(new pcl::PointCloud<pcl::PointXYZI>);
    //   transformLidar(state, p_imu, feats_down_body, world_lidar);
    //   for (size_t i = 0; i < feats_down_body->size(); i++) {
    //     pointWithCov pv;
    //     pv.point << feats_down_body->points[i].x, feats_down_body->points[i].y, feats_down_body->points[i].z;
    //     pv.point_world << world_lidar->points[i].x, world_lidar->points[i].y, world_lidar->points[i].z;
    //     Matrix3 cov            = body_var[i];
    //     Matrix3 point_crossmat = crossmat_list[i];
    //     Matrix3 rot_var        = state.cov.block<3, 3>(0, 0);
    //     Matrix3 t_var          = state.cov.block<3, 3>(3, 3);
    //     cov    = state.rot_end * cov * state.rot_end.transpose() + (-point_crossmat) * rot_var * (-point_crossmat.transpose()) + t_var;
    //     pv.cov = cov;
    //     pv_list.push_back(pv);
    //     var_list.push_back(cov);
    //   }
    //   std::vector<Vector3> non_match_list;
    //   BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, ptpl_list, non_match_list);

    //   effct_feat_num = 0;
    //   for (int i = 0; i < ptpl_list.size(); i++) {
    //     PointType pi_body;
    //     PointType pi_world;
    //     PointType pl;
    //     pi_body.x = ptpl_list[i].point(0);
    //     pi_body.y = ptpl_list[i].point(1);
    //     pi_body.z = ptpl_list[i].point(2);
    //     pointBodyToWorld(&pi_body, &pi_world);
    //     pl.x = ptpl_list[i].normal(0);
    //     pl.y = ptpl_list[i].normal(1);
    //     pl.z = ptpl_list[i].normal(2);
    //     effct_feat_num++;
    //     float dis    = (pi_world.x * pl.x + pi_world.y * pl.y + pi_world.z * pl.z + ptpl_list[i].d);
    //     pl.intensity = dis;
    //     laserCloudOri->push_back(pi_body);
    //     corr_normvect->push_back(pl);
    //   }

    //   /*** Computation of Measuremnt Jacobian matrix H and measurents vector
    //    * ***/
    //   Eigen::MatrixXd Hsub(effct_feat_num, 6);
    //   Eigen::MatrixXd Hsub_T_R_inv(6, effct_feat_num);
    //   Eigen::VectorXd R_inv(effct_feat_num);
    //   Eigen::VectorXd meas_vec(effct_feat_num);

    //   for (int i = 0; i < effct_feat_num; i++) {
    //     const PointType &laser_p = laserCloudOri->points[i];
    //     Vector3 point_this(laser_p.x, laser_p.y, laser_p.z);
    //     Matrix3 cov;
    //     if (calib_laser) {
    //       calcBodyCov(point_this, ranging_cov, CALIB_ANGLE_COV, cov);
    //     } else {
    //       calcBodyCov(point_this, ranging_cov, angle_cov, cov);
    //     }

    //     cov = state.rot_end * cov * state.rot_end.transpose();
    //     Matrix3 point_crossmat;
    //     point_crossmat << SKEW_SYM_MATRX(point_this);
    //     const PointType &norm_p = corr_normvect->points[i];
    //     Vector3 norm_vec(norm_p.x, norm_p.y, norm_p.z);
    //     Vector3 point_world = state.rot_end * point_this + state.pos_end;
    //     // /*** get the normal vector of closest surface/corner ***/
    //     Eigen::Matrix<double, 1, 6> J_nq;
    //     J_nq.block<1, 3>(0, 0)             = point_world - ptpl_list[i].center;
    //     J_nq.block<1, 3>(0, 3)             = -ptpl_list[i].normal;
    //     double sigma_l                     = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
    //     R_inv(i)                           = 1.0 / (sigma_l + norm_vec.transpose() * cov * norm_vec);
    //     double ranging_dis                 = point_this.norm();
    //     laserCloudOri->points[i].intensity = sqrt(R_inv(i));
    //     laserCloudOri->points[i].normal_x  = corr_normvect->points[i].intensity;
    //     laserCloudOri->points[i].normal_y  = sqrt(sigma_l);
    //     laserCloudOri->points[i].normal_z  = sqrt(norm_vec.transpose() * cov * norm_vec);
    //     laserCloudOri->points[i].curvature = sqrt(sigma_l + norm_vec.transpose() * cov * norm_vec);

    //     /*** calculate the Measuremnt Jacobian matrix H ***/
    //     Vector3 A(point_crossmat * state.rot_end.transpose() * norm_vec);
    //     Hsub.row(i) << VEC_FROM_ARRAY(A), norm_p.x, norm_p.y, norm_p.z;
    //     Hsub_T_R_inv.col(i) << A[0] * R_inv(i), A[1] * R_inv(i), A[2] * R_inv(i), norm_p.x * R_inv(i), norm_p.y * R_inv(i), norm_p.z * R_inv(i);
    //     /*** Measuremnt: distance to the closest surface/corner ***/
    //     meas_vec(i) = -norm_p.intensity;
    //   }
    //   Eigen::MatrixXd K(DIM_STATE, effct_feat_num);

    //   EKF_stop_flg      = false;
    //   flg_EKF_converged = false;

    //   Eigen::Matrix<double, DIM_STATE, 1> solution;
    //   Eigen::Matrix<double, DIM_STATE, DIM_STATE> G, H_T_H, I_STATE;
    //   Vector3 rot_add, t_add;

    //   /*** Iterative Kalman Filter Update ***/
    //   if (!flg_EKF_inited) {
    //     std::cout << "||||||||||Initiallizing LiDar||||||||||" << std::endl;
    //     /*** only run in initialization period ***/
    //     Eigen::MatrixXd H_init(Eigen::Matrix<double, 9, DIM_STATE>::Zero());
    //     Eigen::MatrixXd z_init(Eigen::Matrix<double, 9, 1>::Zero());
    //     H_init.block<3, 3>(0, 0)  = Matrix3::Identity();
    //     H_init.block<3, 3>(3, 3)  = Matrix3::Identity();
    //     H_init.block<3, 3>(6, 15) = Matrix3::Identity();
    //     z_init.block<3, 1>(0, 0)  = -Log(state.rot_end);
    //     z_init.block<3, 1>(0, 0)  = -state.pos_end;

    //     auto H_init_T = H_init.transpose();
    //     auto &&K_init = state.cov * H_init_T * (H_init * state.cov * H_init_T + 0.0001 * Eigen::Matrix<double, 9, 9>::Identity()).inverse();
    //     solution      = K_init * z_init;

    //     state.resetpose();
    //     EKF_stop_flg = true;
    //   } else {
    //     auto &&Hsub_T                                   = Hsub.transpose();
    //     H_T_H.block<6, 6>(0, 0)                         = Hsub_T_R_inv * Hsub;
    //     Eigen::Matrix<double, DIM_STATE, DIM_STATE> K_1 = (H_T_H + (state.cov).inverse()).inverse();
    //     K                                               = K_1.block<DIM_STATE, 6>(0, 0) * Hsub_T_R_inv;
    //     auto vec                                        = state_propagat - state;
    //     solution                                        = K * meas_vec + vec - K * Hsub * vec.block<6, 1>(0, 0);

    //     state += solution;

    //     rot_add = solution.block<3, 1>(0, 0);
    //     t_add   = solution.block<3, 1>(3, 0);

    //     if ((rot_add.norm() * 57.3 < 0.01) && (t_add.norm() * 100 < 0.015)) {
    //       flg_EKF_converged = true;
    //     }

    //     deltaR = rot_add.norm() * 57.3;
    //     deltaT = t_add.norm() * 100;
    //   }
    //   /*** Rematch Judgement ***/
    //   nearest_search_en = false;
    //   if (flg_EKF_converged || ((rematch_num == 0) && (iterCount == (NUM_MAX_ITERATIONS - 2)))) {
    //     nearest_search_en = true;
    //     rematch_num++;
    //   }

    //   /*** Convergence Judgements and Covariance Update ***/
    //   if (!EKF_stop_flg && (rematch_num >= 2 || (iterCount == NUM_MAX_ITERATIONS - 1))) {
    //     if (flg_EKF_inited) {
    //       /*** Covariance Update ***/
    //       G.setZero();
    //       G.block<DIM_STATE, 6>(0, 0) = K * Hsub;
    //       state.cov                   = (I_STATE - G) * state.cov;
    //       total_distance += (state.pos_end - position_last).norm();
    //       position_last = state.pos_end;

    //       VD(DIM_STATE) K_sum  = K.rowwise().sum();
    //       VD(DIM_STATE) P_diag = state.cov.diagonal();
    //     }
    //     EKF_stop_flg = true;
    //   }

    //   if (EKF_stop_flg) break;
    // }
  }

 private:
  SensorCalib calib_;
  std::unique_ptr<MsgPackSynchronizer> msg_pack_synchronizer_;
  std::unique_ptr<ImuPreprocess> p_imu{new ImuPreprocess()};

  StatesGroup state;
  bool flg_EKF_inited, flg_EKF_converged, EKF_stop_flg = 0;
};
