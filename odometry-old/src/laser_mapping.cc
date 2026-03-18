// This is an advanced implementation of the algorithm described in the
// following paper:
//   J. Zhang and S. Singh. LOAM: Lidar Odometry and Mapping in Real-time.
//     Robotics: Science and Systems Conference (RSS). Berkeley, CA, July 2014.

// Modifier: Livox               dev@livoxtech.com

// Copyright 2013, Ji Zhang, Carnegie Mellon University
// Further contributions copyright (c) 2016, Southwest Research Institute
// All rights reserved.
//
// Redistribution and use in source and binary forms, with or without
// modification, are permitted provided that the following conditions are met:
//
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its
//    contributors may be used to endorse or promote products derived from this
//    software without specific prior written permission.
//
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
// IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
// LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
// CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
// SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
// CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
// ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
// POSSIBILITY OF SUCH DAMAGE.

#include <fmt/format.h>
#include <geometry_msgs/Vector3.h>
#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <migration/proto_io.h>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <so3_math.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <csignal>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

#include "imu_processing.h"
#include "migration/inc_las_writer.h"
#include "migration/logging.h"
#include "preprocess.h"
#include "sophus/se3.hpp"
#include "voxel_map_util.h"

DEFINE_string(project_dirname, "D:\\slam", "Path to the IMU data file");
DEFINE_string(output_dir, "D:\\slam\\output", "Directory to save output trajectory");
DEFINE_bool(indoor, true, "Set to true for indoor environments");

/*** Time Log Variables ***/
bool runtime_pos_log = false, extrinsic_est_en = true;
/**************************/

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_surf_min = 0;
double lidar_end_time       = 0;
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false;

std::deque<double>                     time_buffer;
std::deque<PointCloudXYZI::Ptr>        lidar_buffer;
std::deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());

V3D    position_last(V3D::Zero());
V3D    Lidar_T_wrt_IMU(V3D::Zero());
M3D    Lidar_R_wrt_IMU(M3D::Identity());
double g_lidar_to_imu_offset = 0.0;  // time offset between lidar and imu

/*** EKF inputs and output ***/
MeasureGroup                                 Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom                                  g_state_point;

std::shared_ptr<Preprocess> p_pre(new Preprocess());
std::shared_ptr<ImuProcess> p_imu(new ImuProcess());

/////////////////// voxel map config begin ///////////////////
double                                   ranging_cov     = 0.05;
double                                   angle_cov       = 0.2;
int                                      max_points_size = 100;  // 50, 100, 200
double                                   max_voxel_size  = 0.5;
std::vector<int>                         layer_size{5, 5, 5, 5, 5};
double                                   min_eigen_value    = 0.01;  // 0.005
double                                   min_plane_likeness = 0.2;
int                                      max_layer          = 2;
std::unordered_map<VoxelLoc, OctoTree *> voxel_map;

// point-plane match infomation list
std::vector<ptpl> ptpl_list;
/////////////////// voxel map config end ///////////////////

void SigHandle(int sig) { flg_exit = true; }

double last_pose_timestamp = -1.0;
void   SaveTraj(std::ofstream &fp, double timestamp, const Eigen::Quaterniond &offset_R_L_I,
                const Eigen::Vector3d &offset_T_L_I, const Eigen::Vector3d &grav, const std::vector<Pose6D> &imu_poses) {
  for (const auto &pose : imu_poses) {
    if (pose.offset_time + timestamp <= last_pose_timestamp) {
      spdlog::warn("skip pose at time because of loop back: {:.6f}", pose.offset_time + timestamp);
      continue;
    }

    last_pose_timestamp = timestamp + pose.offset_time;

    Eigen::Quaterniond rot = Eigen::Quaterniond(pose.rot) * offset_R_L_I;
    Eigen::Vector3d    pos = pose.rot * offset_T_L_I + pose.pos;
    fp << fmt::format("{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}",
                        last_pose_timestamp, pos(0), pos(1), pos(2), rot.x(), rot.y(), rot.z(), rot.w(), grav[0], grav[1],
                        grav[2])
       << std::endl;
  }
}

void CorrectImuPoses(double last_dt, const state_ikfom &state_predict, const state_ikfom &state_update,
                     std::vector<Pose6D> &last_imu_poses) {
  Sophus::SE3d T_imu_prev_to_curr = Sophus::SE3d(state_update.rot.toRotationMatrix(), state_update.pos) *
                                    Sophus::SE3d(state_predict.rot.toRotationMatrix(), state_predict.pos).inverse();
  auto T_imu_prev_to_curr_se3 = T_imu_prev_to_curr.log();
  for (auto &pose : last_imu_poses) {
    double factor = pose.offset_time / last_dt;

    Sophus::SE3d T_correction = Sophus::SE3d::exp(T_imu_prev_to_curr_se3 * factor);

    pose.pos = (T_correction.rotationMatrix() * pose.pos + T_correction.translation()).eval();
    pose.rot = (T_correction.rotationMatrix() * pose.rot).eval();

    spdlog::debug("Correct IMU pose at time: {:.6f} {:.6f} {:.6f}", pose.pos(0), pose.pos(1), pose.pos(2));
  }
}

template <typename T>
void PointBodyToWorld(const Eigen::Map<Eigen::Matrix<T, 3, 1>> &pi, Eigen::Map<Eigen::Matrix<T, 3, 1>> &po,
                      state_ikfom &s) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void livox_pcl_cbk(const std::shared_ptr<proto::LidarMsg> &msg) {
  if (msg->points().size() == 0) {
    spdlog::critical("Empty LiDAR frame at time: {:.6f}", msg->points().at(0).timestamp());
    return;
  }

  double preprocess_start_time = omp_get_wtime();
  if (msg->points().at(0).timestamp() < last_timestamp_lidar) {
    spdlog::error("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->points().at(0).timestamp();

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec() + g_lidar_to_imu_offset);

  double timestamp = msg->header.stamp.toSec();

  if (timestamp < last_timestamp_imu) {
    spdlog::warn("imu loop back, skip data at time: {:.6f}", timestamp);
    return;
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
}

void load_calibration_from_file(const std::string &filename, Eigen::Matrix3d &Lidar_R_wrt_IMU,
                                Eigen::Vector3d &Lidar_T_wrt_IMU, double &lidar_to_imu_offset) {
  if (!boost::filesystem::exists(filename)) {
    spdlog::error("Calibration file does not exist: {}", filename);
    exit(1);
  }

  proto::SensorCalib calib;
  ReadSensorCalibFile(filename, calib);

  // Lidar to IMU
  Lidar_R_wrt_IMU = Eigen::Quaterniond(calib.lidar_to_encoder().rw(), calib.lidar_to_encoder().rx(),
                                       calib.lidar_to_encoder().ry(), calib.lidar_to_encoder().rz())
                        .toRotationMatrix();
  Lidar_T_wrt_IMU << calib.lidar_to_encoder().tx(), calib.lidar_to_encoder().ty(), calib.lidar_to_encoder().tz();
  lidar_to_imu_offset = calib.lidar_to_encoder().time_offset();

  spdlog::info("Loaded calibration from: {}", filename);
  spdlog::info("Lidar_T_wrt_IMU: {:.6f} {:.6f} {:.6f}", Lidar_T_wrt_IMU[0], Lidar_T_wrt_IMU[1], Lidar_T_wrt_IMU[2]);
  spdlog::info("Lidar_R_wrt_IMU: {:.6f} {:.6f} {:.6f} {:.6f}", calib.lidar_to_encoder().rx(),
               calib.lidar_to_encoder().ry(), calib.lidar_to_encoder().rz(), calib.lidar_to_encoder().rw());
  spdlog::info("Lidar to IMU time offset: {:.6f}", lidar_to_imu_offset);
}

// Read data from IMU file and fill into buffer
void load_imu_data_from_file(const std::string &filename) {
  if (!boost::filesystem::exists(filename)) {
    spdlog::error("IMU file does not exist: {}", filename);
    exit(1);
  }

  proto::ImuMsgList imu_msgs;
  ReadImuFile(filename, imu_msgs);
  for (const auto &imu_msg : imu_msgs.imu_msgs()) {
    sensor_msgs::Imu::Ptr imu_ros_msg(new sensor_msgs::Imu());
    imu_ros_msg->header.stamp    = ros::Time(imu_msg.timestamp());
    imu_ros_msg->header.frame_id = "imu_link";
    // todo kk
    imu_ros_msg->linear_acceleration.x = imu_msg.ax();
    imu_ros_msg->linear_acceleration.y = imu_msg.ay();
    imu_ros_msg->linear_acceleration.z = imu_msg.az();
    imu_ros_msg->angular_velocity.x    = imu_msg.gx();
    imu_ros_msg->angular_velocity.y    = imu_msg.gy();
    imu_ros_msg->angular_velocity.z    = imu_msg.gz();
    imu_cbk(imu_ros_msg);
  }

  spdlog::info("Loaded IMU data from: {}, msg count: {}", filename, imu_msgs.imu_msgs_size());
}

double lidar_mean_scantime = 0.0;
int    scan_num            = 0;
bool   sync_packages(MeasureGroup &meas) {
  if (lidar_buffer.empty() || imu_buffer.empty()) {
    return false;
  }

  /*** push a lidar scan ***/
  if (!lidar_pushed) {
    meas.lidar          = lidar_buffer.front();
    meas.lidar_beg_time = time_buffer.front();
    if (meas.lidar->points.size() <= 1)  // time too little
    {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
      spdlog::warn("Too few input point cloud!");
    } else if (meas.lidar->points.back().curvature / double(1000) < 0.5 * lidar_mean_scantime) {
      lidar_end_time = meas.lidar_beg_time + lidar_mean_scantime;
    } else {
      scan_num++;
      lidar_end_time = meas.lidar_beg_time + meas.lidar->points.back().curvature / double(1000);
      lidar_mean_scantime += (meas.lidar->points.back().curvature / double(1000) - lidar_mean_scantime) / scan_num;
    }

    meas.lidar_end_time = lidar_end_time;

    lidar_pushed = true;
  }

  if (last_timestamp_imu < lidar_end_time) {
    return false;
  }

  /*** push imu data, and pop from imu buffer ***/
  double imu_time = imu_buffer.front()->header.stamp.toSec();
  meas.imu.clear();
  while ((!imu_buffer.empty()) && (imu_time < lidar_end_time)) {
    imu_time = imu_buffer.front()->header.stamp.toSec();
    if (imu_time > lidar_end_time) break;
    meas.imu.push_back(imu_buffer.front());
    imu_buffer.pop_front();
  }

  lidar_buffer.pop_front();
  time_buffer.pop_front();
  lidar_pushed = false;
  return true;
}

std::vector<pointWithCov> ComputePvList(const state_ikfom &state) {
  std::vector<pointWithCov> pv_list;
  pv_list.resize(feats_down_world->size());  // 预分配空间，避免并行竞争
#pragma omp parallel for
  for (int i = 0; i < feats_down_world->size(); i++) {
    pointWithCov pv;
    V3D          point_this = feats_down_body->points[i].getVector3fMap().cast<double>();
    CalcBodyCov(point_this, ranging_cov, angle_cov, pv.body_cov);

    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);
    M3D rot_L_W = (state.rot * state.offset_R_L_I).toRotationMatrix();
    V3D pos_L_W = state.rot.toRotationMatrix() * state.offset_T_L_I + state.pos;

    auto &P = kf.get_P();  // must use & here
    pv.pl   = point_this;
    pv.pi   = state.offset_R_L_I * pv.pl + state.offset_T_L_I;
    pv.pw   = state.rot * pv.pi + state.pos;
    pv.cov  = rot_L_W * pv.body_cov * rot_L_W.transpose() +
             rot_L_W * point_crossmat * P.block<3, 3>(3, 3) * point_crossmat.transpose() * rot_L_W.transpose() +
             P.block<3, 3>(0, 0);
    pv_list[i] = pv;  // ✓ 直接赋值，线程安全
  }
  return pv_list;
}

void map_incremental() {
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    PointBodyToWorld(feats_down_body->points[i].getVector3fMap(), feats_down_world->points[i].getVector3fMap(),
                     g_state_point);
    feats_down_world->points[i].intensity = feats_down_body->points[i].intensity;
  }

  // update voxelmap
  std::vector<pointWithCov> pv_list = ComputePvList(g_state_point);
  UpdateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value,
                 voxel_map);
}

// Global variables to store timing for current frame
double g_match_time = 0, g_solve_time = 0;

Eigen::Matrix<double, Eigen::Dynamic, 1> h_share_model(state_ikfom                           &s,
                                                       esekfom::dyn_share_datastruct<double> &ekfom_data) {
  double match_start = omp_get_wtime();

  std::vector<pointWithCov> pv_list = ComputePvList(s);
  std::vector<V3D>          non_match_list;
  /** LiDAR match based on 3 sigma criterion **/
  BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, ptpl_list, non_match_list);

  int effct_feat_num = ptpl_list.size();

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    spdlog::warn("No Effective Points!");
    return {};
  }

  g_match_time        = omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = Eigen::MatrixXd::Zero(effct_feat_num, 12);  // 23
  ekfom_data.h.resize(effct_feat_num);
  ekfom_data.R.resize(effct_feat_num, 1);

  for (int i = 0; i < effct_feat_num; i++) {
    auto &pv = ptpl_list[i].pv;
    V3D   point_this_be(pv.pl);
    M3D   point_be_crossmat;
    point_be_crossmat << SKEW_SYM_MATRX(point_this_be);
    V3D point_this = pv.pi;
    M3D point_crossmat;
    point_crossmat << SKEW_SYM_MATRX(point_this);

    /*** get the normal vector of closest surface/corner ***/
    V3D norm_vec(ptpl_list[i].normal);

    double dist = norm_vec.dot(pv.pw) + ptpl_list[i].d;
    // compute R
    Eigen::Matrix<double, 1, 6> J_nq;
    J_nq.block<1, 3>(0, 0) = pv.pw - ptpl_list[i].center;
    J_nq.block<1, 3>(0, 3) = -norm_vec;
    double sigma_l         = J_nq * ptpl_list[i].plane_cov * J_nq.transpose();
    M3D    rot_L_W         = (s.rot * s.offset_R_L_I).toRotationMatrix();
    ekfom_data.R(i)        = sigma_l + norm_vec.transpose() * rot_L_W * pv.body_cov * rot_L_W.transpose() * norm_vec;

    /*** calculate the Measuremnt Jacobian matrix H ***/
    V3D C(s.rot.conjugate() * norm_vec);
    V3D A(point_crossmat * C);
    if (extrinsic_est_en) {
      V3D B(point_be_crossmat * s.offset_R_L_I.conjugate() * C);  // s.rot.conjugate()*norm_vec);
      ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), VEC_FROM_ARRAY(B),
          VEC_FROM_ARRAY(C);
    } else {
      ekfom_data.h_x.block<1, 12>(i, 0) << VEC_FROM_ARRAY(norm_vec), VEC_FROM_ARRAY(A), 0.0, 0.0, 0.0, 0.0, 0.0, 0.0;
    }

    /*** Measuremnt: distance to the closest surface/corner ***/
    ekfom_data.h(i) = -dist;
  }
  g_solve_time = omp_get_wtime() - solve_start_;

  return ekfom_data.h;
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  InitSpdLog();

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  spdlog::info("Using {}/{} cores.", cores_used, cores);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  if (!std::filesystem::exists(FLAGS_output_dir)) {
    spdlog::info("Creating {}...", FLAGS_output_dir);
    std::filesystem::create_directories(FLAGS_output_dir);
  }

  bool init_map = false;

  NUM_MAX_ITERATIONS   = 4;
  filter_size_surf_min = 0.5;
  gyr_cov              = 0.1;
  acc_cov              = 0.1;
  b_gyr_cov            = 0.0001;
  b_acc_cov            = 0.0001;
  p_pre->blind         = 0.2;
  p_pre->SCAN_RATE     = 10;
  runtime_pos_log      = true;
  extrinsic_est_en     = false;

  // Load calibration parameters from JSON file
  load_calibration_from_file(FLAGS_project_dirname + "/calibration.dat", Lidar_R_wrt_IMU, Lidar_T_wrt_IMU,
                             g_lidar_to_imu_offset);

  ranging_cov = 0.05;
  angle_cov   = 0.2;

  if (FLAGS_indoor) {
    spdlog::info("Indoor mode enabled.");
    max_points_size      = 200;
    max_voxel_size       = 0.5;
    max_layer            = 2;
    layer_size           = std::vector<int>({5, 5, 5, 5, 5});
    filter_size_surf_min = 0.25;
  } else {
    spdlog::info("Outdoor mode enabled.");
    max_points_size      = 200;
    max_voxel_size       = 1.0;
    max_layer            = 2;
    layer_size           = std::vector<int>({5, 5, 5, 5, 5});
    filter_size_surf_min = 0.5;
  }

  min_eigen_value                = 0.01;
  min_plane_likeness             = 0.2;
  int skip_to_save_first_n_scans = 50;

  InitVoxelMapParams(min_plane_likeness);

  /*** variables definition ***/
  int    effect_feat_num = 0, frame_num = 0;
  double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  double epsi[23] = {0.001};
  std::fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

  /*** traj record ***/
  std::string   traj_dir = FLAGS_output_dir + "/traj.txt";
  std::ofstream fp_traj(traj_dir);
  if (!fp_traj) {
    spdlog::error("Failed to open trajectory file: {}", traj_dir);
    return -1;
  }
  fp_traj << "#timestamp_s tx ty tz qx qy qz qw\n";
  fp_traj.flush();

  /*** Load IMU data ***/
  try {
    load_imu_data_from_file(FLAGS_project_dirname + "/imu.dat");
  } catch (const std::exception &e) {
    spdlog::error("Failed to load IMU data: {}", e.what());
    return -1;
  }

  /*** Create LiDAR file reader ***/
  SequentialLidarFileReader<proto::LidarMsg> lidar_reader;
  lidar_reader.Open(FLAGS_project_dirname + "/lidar.dat");
  Ptr<proto::LidarMsg> lidar_msg;

  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::GpsTime);
  table.layout()->registerDim(pdal::Dimension::Id::Intensity);

  std::unique_ptr<migration::IncrementalLasWriter> las_writer = std::make_unique<migration::IncrementalLasWriter>();
  las_writer->initialize(FLAGS_output_dir + "/map.las", table);

  SequentialLidarFileWriter<proto::LidarMsg> lidar_undist_writer;
  lidar_undist_writer.Open(FLAGS_output_dir + "/lidar_undist.dat");

  proto::PoseMsgList traj_dat;

  //------------------------------------------------------------------------------------------------------
  signal(SIGINT, SigHandle);

  double last_timestamp;

  int count = 0;
  // Main processing loop: read and process LiDAR data frame by frame
  while (!lidar_reader.IsFileEnded() && !flg_exit) {
    // Read one frame of point cloud data
    std::shared_ptr<proto::LidarMsg> scan_msg{new proto::LidarMsg()};
    if (lidar_reader.ReadNext(scan_msg)) {
      // Send point cloud data to processing pipeline
      if (scan_msg && scan_msg->points().size() > 0) {
        livox_pcl_cbk(scan_msg);
      }
    }

    // Process synchronized data packages
    if (sync_packages(Measures)) {
      if (flg_first_scan)  // skip the first lidar scan
      {
        flg_first_scan = false;
        continue;
      }

      double t0, t1, t3, t5, svd_time;

      g_match_time = 0;
      g_solve_time = 0;
      svd_time     = 0;
      t0           = omp_get_wtime();

      p_imu->Process(Measures, kf, feats_undistort);
      g_state_point = kf.get_x();

      if (feats_undistort == nullptr || feats_undistort->empty()) {
        spdlog::warn("No point, skip this scan!");
        continue;
      }

      /*** downsample the feature points in a scan ***/
      DownSamplingVoxelRandom<PointType>(*feats_undistort, *feats_down_body, filter_size_surf_min);
      spdlog::info("Downsampled from {} to {} points with voxel size {}.", feats_undistort->points.size(),
                   feats_down_body->points.size(), filter_size_surf_min);
      t1              = omp_get_wtime();
      feats_down_size = feats_down_body->points.size();

      /*** initialize the map kdtree ***/
      if (!init_map) {
        if (feats_down_size > 5) {
          feats_down_world->resize(feats_down_size);
          for (int i = 0; i < feats_down_size; i++) {
            PointBodyToWorld(feats_down_body->points[i].getVector3fMap(), feats_down_world->points[i].getVector3fMap(),
                             g_state_point);
            feats_down_world->points[i].intensity = feats_down_body->points[i].intensity;
          }

          // init voxel map
          std::vector<pointWithCov> pv_list = ComputePvList(g_state_point);
          BuildVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size,
                        min_eigen_value, voxel_map);
          spdlog::info("Initialize voxel map done.");

          init_map = true;
        }
        continue;
      }

      /*** ICP and iterated Kalman filter update ***/
      if (feats_down_size < 5) {
        spdlog::warn("No point, skip this scan!");
        continue;
      }

      feats_down_world->resize(feats_down_size);

      double t_predict_start = omp_get_wtime();

      /*** iterated state estimation ***/
      double solve_H_time  = 0;
      auto   state_predict = kf.get_x();

      double t_predict_end        = omp_get_wtime();
      double t_update_start_local = omp_get_wtime();

      kf.update_iterated_dyn_share_fastlio();
      g_state_point = kf.get_x();

      double t_update_end_local = omp_get_wtime();

      /******* Publish odometry *******/

      /*** add the feature points to map kdtree ***/
      t3 = omp_get_wtime();
      map_incremental();
      t5 = omp_get_wtime();

      /******* Publish points *******/
      if (skip_to_save_first_n_scans-- < 0) {
        {
          int                 size = feats_undistort->points.size();
          PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

          spdlog::info("Adding {} points to the map.", size);

          pdal::PointViewPtr view(new pdal::PointView(table));  // 指定点数
          for (int i = 0; i < size; i++) {
            PointBodyToWorld(feats_undistort->points[i].getVector3fMap(), laserCloudWorld->points[i].getVector3fMap(),
                             g_state_point);
            laserCloudWorld->points[i].intensity = feats_undistort->points[i].intensity;

            view->setField(pdal::Dimension::Id::X, i, laserCloudWorld->points[i].x);
            view->setField(pdal::Dimension::Id::Y, i, laserCloudWorld->points[i].y);
            view->setField(pdal::Dimension::Id::Z, i, laserCloudWorld->points[i].z);
            view->setField(pdal::Dimension::Id::GpsTime, i, Measures.lidar_end_time);
            view->setField(pdal::Dimension::Id::Intensity, i, (uint16_t)laserCloudWorld->points[i].intensity);
          }
          las_writer->writeView(view);
        }

        // Write undistorted lidar scan (body frame, motion-distortion corrected)
        {
          auto undist_msg = std::make_shared<proto::LidarMsg>();
          for (int i = 0; i < (int)feats_undistort->points.size(); i++) {
            auto pt = undist_msg->add_points();
            pt->set_x(feats_undistort->points[i].x);
            pt->set_y(feats_undistort->points[i].y);
            pt->set_z(feats_undistort->points[i].z);
            pt->set_intensity(feats_undistort->points[i].intensity);
            pt->set_timestamp(Measures.lidar_end_time);
          }
          lidar_undist_writer.Write(undist_msg);
        }

        // Write low-frequency pose (one per LiDAR scan)
        {
          auto pose_msg = traj_dat.add_pose_msgs();
          pose_msg->set_timestamp(Measures.lidar_end_time);
          Eigen::Quaterniond q(g_state_point.rot.toRotationMatrix());
          pose_msg->set_tx(g_state_point.pos.x());
          pose_msg->set_ty(g_state_point.pos.y());
          pose_msg->set_tz(g_state_point.pos.z());
          pose_msg->set_rx(q.x());
          pose_msg->set_ry(q.y());
          pose_msg->set_rz(q.z());
          pose_msg->set_rw(q.w());
        }

        auto imu_poses = p_imu->IMUpose;
        if (!imu_poses.empty()) {
          CorrectImuPoses(Measures.lidar_end_time - last_timestamp, state_predict, g_state_point, imu_poses);

          SaveTraj(fp_traj, last_timestamp, g_state_point.offset_R_L_I, g_state_point.offset_T_L_I, g_state_point.grav,
                   imu_poses);
        }
        std::cout << "Progress " << lidar_reader.getProgress() << "%" << std::endl;
      }

      /*** Debug variables ***/
      if (runtime_pos_log) {
        frame_num++;
        double downsample_time  = t1 - t0;
        double match_time_frame = g_match_time;
        double predict_time     = t_predict_end - t_predict_start;
        double update_time      = t_update_end_local - t_update_start_local;
        double map_incr_time    = t5 - t3;
        double total_time       = t5 - t0;

        spdlog::info(
            "[ mapping ]: frame id: {:5d} | Down: {:.6f} Match: {:.6f} Predict: {:.6f} Update: {:.6f} "
            "MapIncr: {:.6f} Total: {:.6f}",
            frame_num, downsample_time, match_time_frame, predict_time, update_time, map_incr_time, total_time);
      }

      last_timestamp = Measures.lidar_end_time;
    }
  }

  las_writer->finalize(table);
  spdlog::info("Finished writing map to LAS file: {}", FLAGS_output_dir + "/map.las");

  lidar_undist_writer.Close();
  spdlog::info("Finished writing undistorted lidar to: {}", FLAGS_output_dir + "/lidar_undist.dat");

  WritePoseFile(FLAGS_output_dir + "/traj.dat", traj_dat);
  spdlog::info("Finished writing trajectory to: {}", FLAGS_output_dir + "/traj.dat");

  /**************** save map ****************/
  // Clean up LAS writer (unique_ptr auto cleanup)
  las_writer.reset();
}
