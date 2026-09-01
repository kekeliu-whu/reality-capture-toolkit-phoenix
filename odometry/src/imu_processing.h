#include <common_lib.h>
#include <geometry_msgs/Vector3.h>
#include <math.h>
#include <nav_msgs/Odometry.h>
#include <pcl/common/io.h>
#include <pcl/common/transforms.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <so3_math.h>
#include <Eigen/Eigen>
#include <cmath>
#include <condition_variable>
#include <csignal>
#include <deque>
#include <fstream>
#include <mutex>
#include <thread>
#include "use-ikfom.h"

struct Pose6D {
  double             offset_time;
  Eigen::Vector3d    acc;
  Eigen::Vector3d    gyr;
  Eigen::Vector3d    vel;
  Eigen::Vector3d    pos;
  Eigen::Matrix3d    rot;
};

struct DynamicInitializationOptions {
  bool   enabled                    = true;
  double min_duration_sec           = 2.0;
  double max_duration_sec           = 4.0;
  double max_acc_std_mps2           = 0.35;
  double max_gyr_std_radps          = 0.08;
  double max_mean_gyr_radps         = 0.08;
  double max_acc_norm_error_mps2    = 0.75;
  double robust_sample_ratio        = 0.25;
};

inline bool time_list(PointType &x, PointType &y) { return (x.curvature < y.curvature); }

/// *************IMU Process and undistortion
class ImuProcess {
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ImuProcess();
  ~ImuProcess();

  void                          Reset();
  void                          Reset(double start_timestamp, const sensor_msgs::ImuConstPtr &lastimu);
  void                          set_extrinsic(const V3D &transl, const M3D &rot);
  void                          set_extrinsic(const V3D &transl);
  void                          set_extrinsic(const MD(4, 4) & T);
  void                          set_gyr_cov(const V3D &scaler);
  void                          set_acc_cov(const V3D &scaler);
  void                          set_gyr_bias_cov(const V3D &b_g);
  void                          set_acc_bias_cov(const V3D &b_a);
  void                          set_dynamic_initialization_options(const DynamicInitializationOptions &options);
  Eigen::Matrix<double, 12, 12> Q;
  bool Process(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
               PointCloudXYZI::Ptr pcl_un);
  bool initialized() const { return !imu_need_init_; }

  V3D            cov_acc;
  V3D            cov_gyr;
  V3D            cov_acc_scale;
  V3D            cov_gyr_scale;
  V3D            cov_bias_gyr;
  V3D            cov_bias_acc;
  double         first_lidar_time;
  std::vector<Pose6D> IMUpose;

 private:
  bool TryInitialize(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state);
  void FinishInitialization(const MeasureGroup &meas,
                            esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                            const V3D &acc_reference, const V3D &gyr_reference,
                            const V3D &acc_std, const V3D &gyr_std, bool stationary);
  void UndistortPcl(const MeasureGroup &meas, esekfom::esekf<state_ikfom, 12, input_ikfom> &kf_state,
                    PointCloudXYZI &pcl_in_out);

  sensor_msgs::ImuConstPtr        last_imu_;
  std::deque<sensor_msgs::ImuConstPtr> v_imu_;
  std::vector<M3D>                     v_rot_pcl_;
  M3D                             Lidar_R_wrt_IMU;
  V3D                             Lidar_T_wrt_IMU;
  V3D                             mean_acc;
  V3D                             mean_gyr;
  V3D                             angvel_last;
  V3D                             acc_s_last;
  double                          start_timestamp_;
  double                          last_lidar_end_time_;
  DynamicInitializationOptions    dynamic_init_options_;
  std::vector<V3D, Eigen::aligned_allocator<V3D>> init_acc_samples_;
  std::vector<V3D, Eigen::aligned_allocator<V3D>> init_gyr_samples_;
  bool                            init_wait_logged_ = false;
  bool                            b_first_frame_ = true;
  bool                            imu_need_init_ = true;
};
