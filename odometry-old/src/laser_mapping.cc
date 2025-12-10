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
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <so3_math.h>
#include <spdlog/spdlog.h>
#include <Eigen/Core>
#include <boost/filesystem.hpp>
#include <csignal>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <memory>
#include <thread>

#include "imu_processing.h"
#include "migration/inc_las_writer.h"
#include "migration/logging.h"
#include "migration/string.h"
#include "preprocess.h"
#include "voxel_map_util.h"

DEFINE_string(imu_filename, UTF8ToPlatform("C:/4.indoor-big-slow/会议室分层/20251023113331_imu.csv"),
              "Path to the IMU data file");
DEFINE_string(lidar_filename, UTF8ToPlatform("C:/4.indoor-big-slow/会议室分层/20251023113331.bin"),
              "Path to the LiDAR data file");
DEFINE_string(output_dir, UTF8ToPlatform("C:/4.indoor-big-slow/会议室分层/out"), "Directory to save output trajectory");

/*** Time Log Variables ***/
double match_time = 0, solve_time = 0, solve_const_H_time = 0;
bool   runtime_pos_log = false, time_sync_en = false, extrinsic_est_en = true;
/**************************/

int point_id = 0;

double last_timestamp_lidar = 0, last_timestamp_imu = -1.0;
double gyr_cov = 0.1, acc_cov = 0.1, b_gyr_cov = 0.0001, b_acc_cov = 0.0001;
double filter_size_corner_min = 0, filter_size_surf_min = 0, filter_size_map_min = 0;
double lidar_end_time = 0, first_lidar_time = 0.0;
int    feats_down_size = 0, NUM_MAX_ITERATIONS = 0;
bool   lidar_pushed, flg_first_scan = true, flg_exit = false;
bool   scan_pub_en = false, dense_pub_en = false, scan_body_pub_en = false;

vector<double>                    extrinT(3, 0.0);
vector<double>                    extrinR(9, 0.0);
deque<double>                     time_buffer;
deque<PointCloudXYZI::Ptr>        lidar_buffer;
deque<sensor_msgs::Imu::ConstPtr> imu_buffer;

PointCloudXYZI::Ptr featsFromMap(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_undistort(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_body(new PointCloudXYZI());
PointCloudXYZI::Ptr feats_down_world(new PointCloudXYZI());
PointCloudXYZI::Ptr _featsArray;

V3D position_last(V3D::Zero());
V3D Lidar_T_wrt_IMU(V3D::Zero());
M3D Lidar_R_wrt_IMU(M3D::Identity());

/*** EKF inputs and output ***/
MeasureGroup                                 Measures;
esekfom::esekf<state_ikfom, 12, input_ikfom> kf;
state_ikfom                                  state_point;
vect3                                        pos_lid;

shared_ptr<Preprocess> p_pre(new Preprocess());
shared_ptr<ImuProcess> p_imu(new ImuProcess());

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

void save_traj(std::ofstream &fp) {
  Eigen::Quaterniond rot = state_point.rot * state_point.offset_R_L_I;
  Eigen::Vector3d    pos = state_point.rot.toRotationMatrix() * state_point.offset_T_L_I + state_point.pos;
  fp << fmt::format("{:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f} {:.12f}\n",
                    Measures.lidar_end_time, pos(0), pos(1), pos(2), rot.x(), rot.y(), rot.z(), rot.w(),
                    state_point.grav[0], state_point.grav[1], state_point.grav[2]);
  fp.flush();
}

void pointBodyToWorld_ikfom(PointType const *const pi, PointType *const po, state_ikfom &s) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(s.rot * (s.offset_R_L_I * p_body + s.offset_T_L_I) + s.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void pointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

template <typename T>
void pointBodyToWorld(const Matrix<T, 3, 1> &pi, Matrix<T, 3, 1> &po) {
  V3D p_body(pi[0], pi[1], pi[2]);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po[0] = p_global(0);
  po[1] = p_global(1);
  po[2] = p_global(2);
}

void RGBpointBodyToWorld(PointType const *const pi, PointType *const po) {
  V3D p_body(pi->x, pi->y, pi->z);
  V3D p_global(state_point.rot * (state_point.offset_R_L_I * p_body + state_point.offset_T_L_I) + state_point.pos);

  po->x         = p_global(0);
  po->y         = p_global(1);
  po->z         = p_global(2);
  po->intensity = pi->intensity;
}

void RGBpointBodyLidarToIMU(PointType const *const pi, PointType *const po) {
  V3D p_body_lidar(pi->x, pi->y, pi->z);
  V3D p_body_imu(state_point.offset_R_L_I * p_body_lidar + state_point.offset_T_L_I);

  po->x         = p_body_imu(0);
  po->y         = p_body_imu(1);
  po->z         = p_body_imu(2);
  po->intensity = pi->intensity;
}

void livox_pcl_cbk(const livox_ros_driver::CustomMsg::ConstPtr &msg) {
  double preprocess_start_time = omp_get_wtime();
  if (msg->header.stamp.toSec() < last_timestamp_lidar) {
    spdlog::error("lidar loop back, clear buffer");
    lidar_buffer.clear();
  }
  last_timestamp_lidar = msg->header.stamp.toSec();

  if (!time_sync_en && abs(last_timestamp_imu - last_timestamp_lidar) > 10.0 && !imu_buffer.empty() &&
      !lidar_buffer.empty()) {
    printf("IMU and LiDAR not Synced, IMU time: %lf, lidar header time: %lf \n", last_timestamp_imu,
           last_timestamp_lidar);
  }

  PointCloudXYZI::Ptr ptr(new PointCloudXYZI());
  p_pre->process(msg, ptr);
  lidar_buffer.push_back(ptr);
  time_buffer.push_back(last_timestamp_lidar);
}

void imu_cbk(const sensor_msgs::Imu::ConstPtr &msg_in) {
  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  msg->header.stamp = ros::Time().fromSec(msg_in->header.stamp.toSec());

  double timestamp = msg->header.stamp.toSec();

  if (timestamp < last_timestamp_imu) {
    spdlog::warn("imu loop back, clear buffer");
    imu_buffer.clear();
  }

  last_timestamp_imu = timestamp;

  imu_buffer.push_back(msg);
}

#pragma pack(push, 1)
struct Point {
  double  timestamp;
  float   x;
  float   y;
  float   z;
  uint8_t intensity;
  uint8_t channel_num;
  uint8_t echo_num;
  uint8_t rest;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ScanData {
  uint64_t point_num;
  Point   *points;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct LidarScan {
  uint64_t scan_data_len;
  ScanData scan_data;
};
#pragma pack(pop)

// LiDAR file reader class
class LidarFileReader {
 private:
  std::ifstream file_;
  std::string   filename_;
  double        last_frame_time_;
  double        frame_interval_;
  bool          is_initialized_;
  bool          file_ended_;
  uint64_t      total_points;
  uint64_t      read_points;

 public:
  LidarFileReader(const std::string &filename, double frame_interval = 0.1)
      : filename_(filename),
        last_frame_time_(0.0),
        frame_interval_(frame_interval),
        is_initialized_(false),
        file_ended_(false),
        total_points(0),
        read_points(0) {
    if (!boost::filesystem::exists(filename)) {
      spdlog::critical("LiDAR file does not exist: {}", filename);
      exit(1);
    }

    file_ = std::ifstream(filename.c_str(), std::ios::binary);
    if (!file_) {
      spdlog::critical("Failed to open LiDAR file: {}", filename);
      exit(1);
    }

    // Skip header line
    int64_t scan_data_len;
    file_.read(reinterpret_cast<char *>(&scan_data_len), sizeof(uint64_t));
    int64_t point_num;
    file_.read(reinterpret_cast<char *>(&point_num), sizeof(uint64_t));
    total_points = point_num;

    spdlog::info("Opened LiDAR file: {}", filename);
  }

  // Read one frame of point cloud data
  // Return value: true if successfully read, false if file ended
  bool readOneScan(livox_ros_driver::CustomMsg::Ptr &scan_msg) {
    if (file_ended_ || !file_) {
      return false;
    }

    scan_msg.reset(new livox_ros_driver::CustomMsg());
    scan_msg->header.frame_id = "lidar_link";
    scan_msg->point_num       = 0;

    while (true) {
      Point pt;
      if (!file_.read(reinterpret_cast<char *>(&pt), sizeof(Point))) {
        file_ended_ = true;
        return false;
      }
      read_points++;

      if (flg_exit) {
        file_ended_ = true;
        return false;
      }

      // Initialize first frame
      if (!is_initialized_) {
        last_frame_time_       = pt.timestamp;
        scan_msg->header.stamp = ros::Time(pt.timestamp);
        scan_msg->timebase     = static_cast<uint64_t>(pt.timestamp * 1e9);
        is_initialized_        = true;
      }

      // Check if current frame time range is exceeded
      if (pt.timestamp - last_frame_time_ >= frame_interval_ && scan_msg->point_num > 0) {
        // Current line is the first point of next frame, need to roll back and process in next call
        // Since file pointer cannot be rolled back to line start, we need to save this point
        // For simplified processing, add this point to next frame
        // Update frame time to this point's time
        last_frame_time_ = pt.timestamp;

        spdlog::info("Read a scan with {} points.", scan_msg->point_num);
        // Return current frame (excluding this new point)
        return scan_msg->point_num > 0;
      }

      // If it's the start of a new frame (point count is 0 and initialized)
      if (scan_msg->point_num == 0 && is_initialized_) {
        scan_msg->header.stamp = ros::Time(last_frame_time_);
        scan_msg->timebase     = static_cast<uint64_t>(last_frame_time_ * 1e9);
      }

      // Add point to current frame
      livox_ros_driver::CustomPoint point;
      point.x            = pt.x;
      point.y            = pt.y;
      point.z            = pt.z;
      point.reflectivity = static_cast<uint8_t>(pt.intensity);
      point.offset_time  = static_cast<uint32_t>((pt.timestamp - scan_msg->header.stamp.toSec()) * 1e9);
      point.line         = pt.channel_num;
      point.tag          = 0;

      scan_msg->points.push_back(point);
      scan_msg->point_num++;
    }

    // File ended, return last frame (if there is data)
    return scan_msg->point_num > 0;
  }

  bool isFileEnded() const { return file_ended_; }

  void setFrameInterval(double interval) { frame_interval_ = interval; }

  double getProgress() const { return total_points > 0 ? (read_points / (double)total_points) * 100.0 : 0.0; }
};

// Read data from IMU file and fill into buffer
void load_imu_data_from_file(const std::string &filename) {
  if (!boost::filesystem::exists(filename)) {
    spdlog::error("IMU file does not exist: {}", filename);
    throw std::runtime_error("IMU file not found");
  }

  std::ifstream imu_file(filename);
  if (!imu_file) {
    spdlog::error("Failed to open IMU file: {}", filename);
    throw std::runtime_error("Failed to open IMU file");
  }

  // Skip header line
  std::string header;
  std::getline(imu_file, header);

  double last_ax = 0, last_ay = 0, last_az = 0;
  bool   is_first_imu = true;

  double t, ax, ay, az, gx, gy, gz;
  char   comma;
  while (imu_file >> t >> comma >> ax >> comma >> ay >> comma >> az >> comma >> gx >> comma >> gy >> comma >> gz) {
    // Skip duplicate IMU data
    if (!is_first_imu && ax == last_ax && ay == last_ay && az == last_az) {
      continue;
    }

    last_ax      = ax;
    last_ay      = ay;
    last_az      = az;
    is_first_imu = false;

    sensor_msgs::Imu::Ptr imu_msg(new sensor_msgs::Imu());
    imu_msg->header.stamp          = ros::Time(t);
    imu_msg->header.frame_id       = "imu_link";
    imu_msg->linear_acceleration.x = ax;
    imu_msg->linear_acceleration.y = ay;
    imu_msg->linear_acceleration.z = az;
    imu_msg->angular_velocity.x    = gx;
    imu_msg->angular_velocity.y    = gy;
    imu_msg->angular_velocity.z    = gz;

    imu_cbk(imu_msg);
  }

  spdlog::info("Loaded IMU data from: {}", filename);
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
  for (size_t i = 0; i < feats_down_world->size(); i++) {
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
    pv_list.push_back(pv);
  }
  return pv_list;
}

int  process_increments = 0;
void map_incremental() {
  for (int i = 0; i < feats_down_size; i++) {
    /* transform to world frame */
    pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
  }

  // update voxelmap
  std::vector<pointWithCov> pv_list = ComputePvList(state_point);
  UpdateVoxelMap(pv_list, max_voxel_size, max_layer, layer_size, max_points_size, max_points_size, min_eigen_value,
                 voxel_map);
}

Matrix<double, Eigen::Dynamic, 1> h_share_model(state_ikfom &s, esekfom::dyn_share_datastruct<double> &ekfom_data) {
  double match_start = omp_get_wtime();

  vector<pointWithCov> pv_list = ComputePvList(s);
  std::vector<V3D>     non_match_list;
  /** LiDAR match based on 3 sigma criterion **/
  BuildResidualListOMP(voxel_map, max_voxel_size, 3.0, max_layer, pv_list, ptpl_list, non_match_list);

  int effct_feat_num = ptpl_list.size();

  if (effct_feat_num < 1) {
    ekfom_data.valid = false;
    spdlog::warn("No Effective Points!");
    return {};
  }

  match_time += omp_get_wtime() - match_start;
  double solve_start_ = omp_get_wtime();

  /*** Computation of Measuremnt Jacobian matrix H and measurents vector ***/
  ekfom_data.h_x = MatrixXd::Zero(effct_feat_num, 12);  // 23
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
  solve_time += omp_get_wtime() - solve_start_;

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

  scan_pub_en            = true;
  dense_pub_en           = true;
  scan_body_pub_en       = true;
  NUM_MAX_ITERATIONS     = 4;
  time_sync_en           = false;
  filter_size_corner_min = 0.5;
  filter_size_surf_min   = 0.5;
  filter_size_map_min    = 0.5;
  gyr_cov                = 0.1;
  acc_cov                = 0.1;
  b_gyr_cov              = 0.0001;
  b_acc_cov              = 0.0001;
  p_pre->blind           = 0.2;
  p_pre->SCAN_RATE       = 10;
  runtime_pos_log        = true;
  extrinsic_est_en       = false;
  extrinT                = vector<double>({-0.027100, 0.021396, -0.009847});
  extrinR =
      vector<double>({0.999938, 0.007471, 0.008288, -0.007465, 0.999972, -0.000677, -0.008293, 0.000615, 0.999965});
  ranging_cov            = 0.05;
  angle_cov              = 0.2;
  max_points_size        = 400;
  max_voxel_size         = 0.5;
  max_layer              = 2;
  layer_size             = vector<int>({5, 5, 5, 5, 5});
  min_eigen_value        = 0.01;
  min_plane_likeness     = 0.2;
  int skip_first_n_scans = 50;

  InitVoxelMapParams(min_plane_likeness);

  /*** variables definition ***/
  int    effect_feat_num = 0, frame_num = 0;
  double aver_time_consu = 0, aver_time_icp = 0, aver_time_match = 0, aver_time_solve = 0, aver_time_const_H_time = 0;

  _featsArray.reset(new PointCloudXYZI());

  Lidar_T_wrt_IMU << VEC_FROM_ARRAY(extrinT);
  Lidar_R_wrt_IMU << MAT_FROM_ARRAY(extrinR);
  p_imu->set_extrinsic(Lidar_T_wrt_IMU, Lidar_R_wrt_IMU);
  p_imu->set_gyr_cov(V3D(gyr_cov, gyr_cov, gyr_cov));
  p_imu->set_acc_cov(V3D(acc_cov, acc_cov, acc_cov));
  p_imu->set_gyr_bias_cov(V3D(b_gyr_cov, b_gyr_cov, b_gyr_cov));
  p_imu->set_acc_bias_cov(V3D(b_acc_cov, b_acc_cov, b_acc_cov));

  double epsi[23] = {0.001};
  fill(epsi, epsi + 23, 0.001);
  kf.init_dyn_share(get_f, df_dx, df_dw, h_share_model, NUM_MAX_ITERATIONS, epsi);

  /*** traj record ***/
  string        traj_dir = FLAGS_output_dir + "/traj.txt";
  std::ofstream fp_traj(traj_dir);
  if (!fp_traj) {
    spdlog::error("Failed to open trajectory file: {}", traj_dir);
    return -1;
  }
  fp_traj << "#timestamp_s tx ty tz qx qy qz qw\n";
  fp_traj.flush();

  /*** Load IMU data ***/
  try {
    load_imu_data_from_file(FLAGS_imu_filename);
  } catch (const std::exception &e) {
    spdlog::error("Failed to load IMU data: {}", e.what());
    return -1;
  }

  /*** Create LiDAR file reader ***/
  std::unique_ptr<LidarFileReader> lidar_reader;
  try {
    lidar_reader = std::make_unique<LidarFileReader>(FLAGS_lidar_filename, 0.1);  // 0.1 seconds per frame
  } catch (const std::exception &e) {
    spdlog::error("Failed to create LiDAR reader: {}", e.what());
    return -1;
  }

  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::GpsTime);
  table.layout()->registerDim(pdal::Dimension::Id::Intensity);

  std::unique_ptr<migration::IncrementalLasWriter> las_writer = std::make_unique<migration::IncrementalLasWriter>();
  las_writer->initialize(FLAGS_output_dir + "/map.las", table);

  //------------------------------------------------------------------------------------------------------
  signal(SIGINT, SigHandle);

  int count = 0;
  // Main processing loop: read and process LiDAR data frame by frame
  while (!lidar_reader->isFileEnded() && !flg_exit) {
    // Read one frame of point cloud data
    livox_ros_driver::CustomMsg::Ptr scan_msg;
    if (lidar_reader->readOneScan(scan_msg)) {
      // Send point cloud data to processing pipeline
      if (scan_msg && scan_msg->point_num > 0) {
        livox_pcl_cbk(scan_msg);
      }
    }

    // Process synchronized data packages
    if (sync_packages(Measures)) {
      if (flg_first_scan)  // skip the first lidar scan
      {
        first_lidar_time = Measures.lidar_beg_time;
        flg_first_scan   = false;
        continue;
      }

      double t0, t1, t2, t3, t5, svd_time;

      match_time         = 0;
      solve_time         = 0;
      solve_const_H_time = 0;
      svd_time           = 0;
      t0                 = omp_get_wtime();

      p_imu->Process(Measures, kf, feats_undistort);
      state_point = kf.get_x();
      pos_lid     = state_point.pos + state_point.rot.toRotationMatrix() * state_point.offset_T_L_I;

      if (feats_undistort == nullptr || feats_undistort->empty()) {
        spdlog::warn("No point, skip this scan!");
        continue;
      }

      /*** downsample the feature points in a scan ***/
      DownSamplingVoxelRandom<PointType>(*feats_undistort, *feats_down_body, filter_size_surf_min);
      t1              = omp_get_wtime();
      feats_down_size = feats_down_body->points.size();

      /*** initialize the map kdtree ***/
      if (!init_map) {
        if (feats_down_size > 5) {
          feats_down_world->resize(feats_down_size);
          for (int i = 0; i < feats_down_size; i++) {
            pointBodyToWorld(&(feats_down_body->points[i]), &(feats_down_world->points[i]));
          }

          // init voxel map
          std::vector<pointWithCov> pv_list = ComputePvList(state_point);
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

      t2 = omp_get_wtime();

      /*** iterated state estimation ***/
      double t_update_start = omp_get_wtime();
      double solve_H_time   = 0;
      kf.update_iterated_dyn_share_fastlio();
      state_point = kf.get_x();
      pos_lid     = state_point.pos + state_point.rot.toRotationMatrix() * state_point.offset_T_L_I;

      double t_update_end = omp_get_wtime();

      /******* Publish odometry *******/

      /*** add the feature points to map kdtree ***/
      t3 = omp_get_wtime();
      map_incremental();
      t5 = omp_get_wtime();

      /******* Publish points *******/
      if (skip_first_n_scans-- < 0) {
        {
          int                 size = feats_undistort->points.size();
          PointCloudXYZI::Ptr laserCloudWorld(new PointCloudXYZI(size, 1));

          spdlog::info("Adding {} points to the map.", size);

          pdal::PointViewPtr view(new pdal::PointView(table));  // 指定点数
          for (int i = 0; i < size; i++) {
            RGBpointBodyToWorld(&feats_undistort->points[i], &laserCloudWorld->points[i]);

            view->setField(pdal::Dimension::Id::X, i, laserCloudWorld->points[i].x);
            view->setField(pdal::Dimension::Id::Y, i, laserCloudWorld->points[i].y);
            view->setField(pdal::Dimension::Id::Z, i, laserCloudWorld->points[i].z);
            view->setField(pdal::Dimension::Id::GpsTime, i, Measures.lidar_end_time);
            view->setField(pdal::Dimension::Id::Intensity, i, (uint16_t)laserCloudWorld->points[i].intensity);
          }
          las_writer->writeView(view);
        }

        save_traj(fp_traj);
        std::cout << "Progress " << lidar_reader->getProgress() << std::endl;
      }

      /*** Debug variables ***/
      if (runtime_pos_log) {
        frame_num++;
        aver_time_consu = aver_time_consu * (frame_num - 1) / frame_num + (t5 - t0) / frame_num;
        aver_time_icp   = aver_time_icp * (frame_num - 1) / frame_num + (t_update_end - t_update_start) / frame_num;
        aver_time_match = aver_time_match * (frame_num - 1) / frame_num + (match_time) / frame_num;
        aver_time_solve = aver_time_solve * (frame_num - 1) / frame_num + (solve_time + solve_H_time) / frame_num;
        aver_time_const_H_time = aver_time_const_H_time * (frame_num - 1) / frame_num + solve_time / frame_num;

        spdlog::info(
            "[ mapping ]: frame id: {:5d} time: IMU + Map + Input Downsample: {:.6f} ave match: {:.6f} ave solve: "
            "{:.6f} "
            "ave ICP: {:.6f} map incre: {:.6f} ave total: {:.6f} icp: {:.6f} construct H: {:.6f}",
            frame_num, t1 - t0, aver_time_match, aver_time_solve, t3 - t1, t5 - t3, aver_time_consu, aver_time_icp,
            aver_time_const_H_time);
      }
    }
  }

  las_writer->finalize(table);
  spdlog::info("Finished writing map to LAS file: {}", FLAGS_output_dir + "/map.las");

  // Clean up resources (unique_ptr auto cleanup)
  lidar_reader.reset();

  /**************** save map ****************/
  // Clean up LAS writer (unique_ptr auto cleanup)
  las_writer.reset();
}
