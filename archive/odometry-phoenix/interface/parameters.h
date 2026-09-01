#pragma once

#include <eigen3/Eigen/Dense>

namespace lixel
{

struct SensorParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct ImuParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Vector3d Ba;
    Eigen::Vector3d Bg;
    Eigen::Matrix3d Ka;
    Eigen::Matrix3d Kg;
    Eigen::Matrix3d Ta;
    Eigen::Matrix3d Tg;
    // Convert incoming IMU-clock timestamps to the LiDAR clock.  Constant
    // offset comes from calibration; drift is an explicit dataset setting.
    double lidar_to_imu_time_offset_seconds = 0.0;
    double clock_drift_ppm = 0.0;
    bool calibrated = false;
    bool enabled = false;
  };

  struct LidarParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    double dist = 3.0;
    double angle = 0.009;
    double elevation_offset = 0.0;
    bool calibrated = false;
    bool enabled = false;
  };

  ImuParam imu_param;
  LidarParam lidar_param;
};

struct ExtrinsicParam
{
  struct MotorParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
    Eigen::Matrix4d ext_motor_lidar;
    bool calibrated = false;
    bool enabled = false;
  };

  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  MotorParam motor_param;
  Eigen::Matrix4d ext_imu_motor;
  Eigen::Vector3d t_imu_gnss;
};

struct ScenarioParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool with_normals = false;
};

struct PreprocessParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double range_min = 0.3;
  double range_max = 120.0;
  Eigen::Vector3d body_mask_min = {0.0, 0.0, 0.0};
  Eigen::Vector3d body_mask_max = {0.0, 0.0, 0.0};
  double sweep_duration = 0.2;
  bool sweep_cut_auto = true;
};

enum struct MapType
{
  XIVT = 0,
  VOXEL_MAP = 1,
  XMAP = 2
};

struct MapParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  struct XivtParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct VoxelMapParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  struct XmapParam
  {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  };

  MapType map_type = MapType::XIVT;
  XivtParam xivt_param;
  XivtParam voxelmap_param;
  XivtParam xmap_param;
  std::string config_path;
};

struct LidarFeatureParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double downsample_size = 0.2;
  double multi_frame = 0;
};
enum SufaceAreaMethod
{
  cube_4_side = 0,  // use cube's smallest 4 faces area
  ellipsoid = 1,    // use all ellipsoid area method
};

struct DownsampleParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  SufaceAreaMethod area_method;
  float init_pca_downsample_dis;
  float base_downsample_dis;
  float max_downsample_dis;
  uint32_t ref_downsample_point_num;
};

struct InitParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  double init_pos_std;         // unit: m
  double init_vel_std;         // unit: m/s
  double init_rot_std;         // unit: rad
  double init_acc_bias_std;    // unit: m/s^3
  double init_gyro_bias_std;   // unit: rad/s
  bool use_initial_pose = false;
  Eigen::Vector4d initial_quaternion_xyzw = {0.0, 0.0, 0.0, 1.0};
  Eigen::Vector3d initial_velocity = {0.0, 0.0, 0.0};

  // Motion-aware LiDAR/IMU bootstrap.  The bootstrap registers consecutive
  // scans in a temporary local frame, estimates the terminal velocity and
  // builds an initialization map with a separate pose for every scan.
  bool dynamic_init_enabled = true;
  double dynamic_init_min_duration = 2.0;
  double dynamic_init_max_duration = 4.0;
  double dynamic_init_icp_voxel_size = 0.25;
  double dynamic_init_icp_max_correspondence = 1.5;
  double dynamic_init_icp_fitness_threshold = 0.35;
  int dynamic_init_min_registration_points = 100;
  double dynamic_init_min_registration_ratio = 0.6;
  double dynamic_init_max_acc_std = 0.35;
  double dynamic_init_max_gyr_std = 0.08;
  double dynamic_init_max_mean_gyr = 0.08;
  double dynamic_init_max_acc_norm_error = 0.75;
  double dynamic_init_robust_sample_ratio = 0.25;
};

struct IESKFParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  int window_size;
  int reset_window_size;
  double acc_std;              // m/s^2
  double gyr_std;              // rad/s
  double acc_std_slope;
  double gyro_std_slope;
  double acc_bias_std;         // m/s^3
  double gyr_bias_std;         // rad/s^2
  int max_iter;                //
  bool faster_model;
  double knn_search_slope;
  double knn_search_min_dist;
  double lidar_std_dev_limit;
  bool use_gnss = false;
  bool use_vio = false;
  bool use_edge = false;
};

struct PerformanceParam
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
  bool use_randofilter = true;
  int max_effect_points_size = 5000;
  int worker_threads = -1;
  int reverse_bind = 0;
  int bind_server = 0;
};

struct LioParameters
{
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  SensorParam sensor_param;
  ExtrinsicParam extrinsic_param;
  LidarFeatureParam lidar_feature_param;
  ScenarioParam scenario_param;
  PreprocessParam preprocess_param;
  MapParam map_param;
  InitParam init_param;
  IESKFParam kf_param;
  DownsampleParam downsample_param;
  PerformanceParam performance_param;
};

}  // namespace lixel
