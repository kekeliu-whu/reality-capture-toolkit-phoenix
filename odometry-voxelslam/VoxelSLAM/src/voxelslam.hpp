#pragma once
#include "platform_compat.hpp"

#include "tools.hpp"
#include "ekf_imu.hpp"
#include "voxel_map.hpp"
#include "feature_point.hpp"
#include "loop_refine.hpp"
#include <mutex>
#include <Eigen/Eigenvalues>
#include <pcl/kdtree/kdtree_flann.h>
#include <gtsam/inference/Symbol.h>
#include <gtsam/navigation/ImuFactor.h>
#include <gtsam/navigation/CombinedImuFactor.h>
#include <gtsam/nonlinear/GaussNewtonOptimizer.h>
#include <gtsam/nonlinear/LevenbergMarquardtOptimizer.h>
#include <Eigen/Sparse>
#include <Eigen/SparseQR>
#include "BTC.h"
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <yaml-cpp/yaml.h>
#include <chrono>

using namespace std;

// ===== Config struct (replaces ros::NodeHandle + ros::param) =====
struct VoxelConfig {
  // General
  string lid_topic = "/livox/lidar";
  string imu_topic = "/livox/imu";
  string save_path = "/home/rick/catkin_ws/data/save_maps/";
  string bagname = "bag_full";
  int lidar_type = 0;
  double blind = 0.5;
  int point_filter_num = 3;
  vector<double> extrinsic_tran = {-0.011, -0.02329, 0.04412};
  vector<double> extrinsic_rota = {1,0,0, 0,1,0, 0,0,1};
  int is_save_map = 1;
  string previous_map = "";

  // Odometry
  double odom_cov_gyr = 0.1;
  double odom_cov_acc = 1.0;
  double odom_rdw_gyr = 0.0001;
  double odom_rdw_acc = 0.0001;
  double down_size = 0.1;
  double dept_err = 0.02;
  double beam_err = 0.05;
  double voxel_size = 1.0;
  double min_eigen_value = 0.0025;
  int degrade_bound = 10;
  int point_notime = 0;

  // LocalBA
  int win_size = 10;
  int max_layer = 2;
  double lba_cov_gyr = 0.01;
  double lba_cov_acc = 1.0;
  double lba_rdw_gyr = 0.0001;
  double lba_rdw_acc = 0.0001;
  int min_ba_point = 1;
  vector<double> plane_eigen_value_thre = {4.0, 4.0, 4.0, 4.0};
  double imu_coef = 0.0001;
  int thread_num = 5;

  // Loop
  double jud_default = 0.5;
  double icp_eigval = 10.0;
  double ratio_drift = 0.01;
  int curr_halt = 10;
  int prev_halt = 10;
  int acsize = 2;
  int mgsize = 2;
  int isHighFly = 0;

  // GBA
  double gba_voxel_size = 2.0;
  double gba_min_eigen_value = 0.01;
  vector<double> gba_eigen_value_array = {4.0, 4.0, 4.0, 4.0};
  int total_max_iter = 6;

  void loadFromYAML(const string &yaml_path);
};

// ===== Offline bag processing (streaming) =====
bool offline_mode = false;
string offline_bag_path;
rosbag::Bag *offline_bag = nullptr;
bool offline_eof = false;
size_t offline_lidar_count = 0;

void open_bag_offline(const string &bag_path);
void close_bag_offline();
bool offline_spin_once();

// Publishers removed (no ROS pub/sub). All pub_pl_func calls become no-ops.
template <typename T>
void pub_pl_func(T &pl, int) { /* no-op: offline mode, no ROS publishing */ }

mutex mBuf;
Features feat;
deque<sensor_msgs::Imu::Ptr> imu_buf;
deque<pcl::PointCloud<PointType>::Ptr> pcl_buf;
deque<double> time_buf;

double imu_last_time = -1;
int point_notime = 0;
double last_pcl_time = -1;

void imu_handler(const sensor_msgs::Imu::ConstPtr &msg_in)
{
  static int flag = 1;
  if(flag)
  {
    flag = 0;
    printf("Time0: %lf\n", msg_in->header.stamp.toSec());
  }

  sensor_msgs::Imu::Ptr msg(new sensor_msgs::Imu(*msg_in));

  // For Hilti 2022 exp03
  // double t0 = 1646320760 + 255.5;
  // double t1 = 1646320760 + 256.2;
  // double tc = msg->header.stamp.toSec();
  // if(tc > t0 && tc < t1)
  //   msg->linear_acceleration.z = -9.7;

  mBuf.lock();
  imu_last_time = msg->header.stamp.toSec();
  imu_buf.push_back(msg);
  mBuf.unlock();
}

template<class T>
void pcl_handler(T &msg)
{
  pcl::PointCloud<PointType>::Ptr pl_ptr(new pcl::PointCloud<PointType>());
  double t0 = feat.process(msg, *pl_ptr);

  if(pl_ptr->empty())
  {
    PointType ap; 
    ap.x = 0; ap.y = 0; ap.z = 0; 
    ap.intensity = 0; ap.curvature = 0;
    pl_ptr->push_back(ap);
    ap.curvature = 0.09;
    pl_ptr->push_back(ap);
  }

  sort(pl_ptr->begin(), pl_ptr->end(), [](PointType &x, PointType &y)
  {
    return x.curvature < y.curvature;
  });
  while(pl_ptr->back().curvature > 0.11)
    pl_ptr->points.pop_back();

  mBuf.lock();
  time_buf.push_back(t0);
  pcl_buf.push_back(pl_ptr);
  mBuf.unlock();
}

static int g_sync_cnt = 0;
bool sync_packages(pcl::PointCloud<PointType>::Ptr &pl_ptr, deque<sensor_msgs::Imu::Ptr> &imus, IMUEKF &p_imu)
{
  static bool pl_ready = false;

  if(!pl_ready)
  {
    if(pcl_buf.empty()) return false;

    mBuf.lock();
    pl_ptr = pcl_buf.front();
    p_imu.pcl_beg_time = time_buf.front();
    pcl_buf.pop_front(); time_buf.pop_front();
    mBuf.unlock();

    p_imu.pcl_end_time = p_imu.pcl_beg_time + pl_ptr->back().curvature;

    if(point_notime)
    {
      if(last_pcl_time < 0)
      {
        last_pcl_time = p_imu.pcl_beg_time;
        return false;
      }

      p_imu.pcl_end_time = p_imu.pcl_beg_time;
      p_imu.pcl_beg_time = last_pcl_time;
      last_pcl_time = p_imu.pcl_end_time;
    }

    pl_ready = true;
  }

  if(!pl_ready || imu_last_time <= p_imu.pcl_end_time) return false;

  mBuf.lock();
  double imu_time = imu_buf.front()->header.stamp.toSec();
  while((!imu_buf.empty()) && (imu_time < p_imu.pcl_end_time))
  {
    imu_time = imu_buf.front()->header.stamp.toSec();
    if(imu_time > p_imu.pcl_end_time) break;
    imus.push_back(imu_buf.front());
    imu_buf.pop_front();
  }
  mBuf.unlock();

  if(imu_buf.empty())
  {
    printf("imu buf empty\n"); exit(0);
  }

  pl_ready = false;

  g_sync_cnt++;
  if(g_sync_cnt % 500 == 0)
    printf("[SYNC] %d packages synced, imus=%zu pcl_buf=%zu\n", g_sync_cnt, imus.size(), pcl_buf.size());

  if(imus.size() > 4)
    return true;
  else
    return false;
}

double dept_err, beam_err;
void calcBodyVar(Eigen::Vector3d &pb, const float range_inc, const float degree_inc, Eigen::Matrix3d &var) 
{
  if (pb[2] == 0)
    pb[2] = 0.0001;
  float range = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0, pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0, -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1, -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1), base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  var = direction * range_var * direction.transpose() + A * direction_var * A.transpose();
};

// Compute the variance of the each point
void var_init(IMUST &ext, pcl::PointCloud<PointType> &pl_cur, PVecPtr pptr, double dept_err, double beam_err)
{
  int plsize = pl_cur.size();
  pptr->clear();
  pptr->resize(plsize);
  for(int i=0; i<plsize; i++)
  {
    PointType &ap = pl_cur[i];
    pointVar &pv = pptr->at(i);
    pv.pnt << ap.x, ap.y, ap.z;
    calcBodyVar(pv.pnt, dept_err, beam_err, pv.var);
    pv.pnt = ext.R * pv.pnt + ext.p;
    pv.var = ext.R * pv.var * ext.R.transpose();
  }
}

void pvec_update(PVecPtr pptr, IMUST &x_curr, PLV(3) &pwld)
{
  Eigen::Matrix3d rot_var = x_curr.cov.block<3, 3>(0, 0);
  Eigen::Matrix3d tsl_var = x_curr.cov.block<3, 3>(3, 3);

  for(pointVar &pv: *pptr)
  {
    Eigen::Matrix3d phat = hat(pv.pnt);
    pv.var = x_curr.R * pv.var * x_curr.R.transpose() + phat * rot_var * phat.transpose() + tsl_var;
    pwld.push_back(x_curr.R * pv.pnt + x_curr.p);
  }
}

// Read the alidarstate.txt
void read_lidarstate(string filename, vector<ScanPose*> &bl_tem)
{
  ifstream file(filename);
  if(!file.is_open())
  {
    printf("Error: %s not found\n", filename.c_str());
    exit(0);
  }

  string lineStr, str;
  vector<double> nums;
  while(getline(file, lineStr))
  {
    nums.clear();
    stringstream ss(lineStr);
    while(getline(ss, str, ' '))
      nums.push_back(stod(str));
    
    IMUST xx;
    xx.t = nums[0];
    xx.p << nums[1], nums[2], nums[3];
    xx.R = Eigen::Quaterniond(nums[7], nums[4], nums[5], nums[6]).matrix();

    if(nums.size() >= 20)
    {
      xx.v << nums[8], nums[9], nums[10];
      xx.bg << nums[11], nums[12], nums[13];
      xx.ba << nums[14], nums[15], nums[16];
      xx.g << nums[17], nums[18], nums[19];
    }

    ScanPose* blp = new ScanPose(xx, nullptr);
    bl_tem.push_back(blp);

    if(nums.size() >= 26)
      for(int i=0; i<6; i++) 
        blp->v6[i] = nums[i + 20];
  }
}

double get_memory()
{
  return platform_get_memory_gb();
}

void icp_check(pcl::PointCloud<PointType> &pl_src, pcl::PointCloud<PointType> &pl_tar, int pub_src, int pub_tar, pair<Eigen::Vector3d, Eigen::Matrix3d> &loop_transform, IMUST &xx)
{
  pcl::PointCloud<PointType> pl1, pl2;
  for(PointType ap: pl_src.points)
  {
    Eigen::Vector3d v(ap.x, ap.y, ap.z);
    v = loop_transform.second * v + loop_transform.first;
    v = xx.R * v + xx.p;
    ap.x = v[0]; ap.y = v[1]; ap.z = v[2];
    pl1.push_back(ap);
  }
  for(PointType ap: pl_tar.points)
  {
    Eigen::Vector3d v(ap.x, ap.y, ap.z);
    v = xx.R * v + xx.p;
    ap.x = v[0]; ap.y = v[1]; ap.z = v[2];
    pl2.push_back(ap);
  }
  pub_pl_func(pl1, pub_src); pub_pl_func(pl2, pub_tar);
}

