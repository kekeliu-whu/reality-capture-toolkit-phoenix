#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include "../../lio_core/thirdparty/sophus/se3.hpp"

typedef pcl::PointXYZI PointT;
typedef pcl::PointCloud<PointT> PointCloud;
typedef pcl::PointXYZRGB PointTRGB;
typedef pcl::PointCloud<PointTRGB> PointCloudRGB;
typedef Sophus::SE3d SE3;

typedef enum
{
  CP_TYPE_LABEL_ONLY = 100,       // 只包含标签的控制点
  CP_TYPE_LABEL_XYZ = 101,        // 包含标签和位置的控制点
  CP_TYPE_LABEL_XYZRPY = 102,     // 包含标签、位置和姿态的控制点
  CP_TYPE_VISUAL = 103,           // 视觉控制点
  CP_TYPE_MULTIFLOOR = 104,       // 多楼层控制点
  CP_TYPE_SAMEPLANE = 105,        // 共面控制点
  CP_TYPE_PARALLEL_PLANE = 106,   // 平行面控制点
  CP_TYPE_ORTHOGONAL_PLANE = 107  // 正交面控制点
} CONTROL_POINT_TYPE;

struct LioState
{
  double timestamp;
  Eigen::Vector3d p;
  Eigen::Quaterniond q;
  Eigen::Vector3d v;
  Eigen::Vector3d ba;
  Eigen::Vector3d bg;
  Eigen::Vector3d grav;
};

class LidarFrame
{
 public:
  using Ptr = std::shared_ptr<LidarFrame>;
  using ConstPtr = std::shared_ptr<const LidarFrame>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  LidarFrame() = default;

  int m_id;
  double m_timestamp;
  SE3 m_pose;
  SE3 m_pose_opt;
  Eigen::Vector3d m_gravity_vec;
  PointCloud::Ptr m_pointcloud;
};

class ControlPoint
{
 public:
  using Ptr = std::shared_ptr<ControlPoint>;
  using ConstPtr = std::shared_ptr<const ControlPoint>;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  ControlPoint()
  {
  }

  ControlPoint(int id, std::string label, CONTROL_POINT_TYPE type)
  {
    m_id = id;
    m_label = label;
    m_type = type;

    m_position = Eigen::Vector3d(NAN, NAN, NAN);
    m_orientation = Eigen::Quaterniond(NAN, NAN, NAN, NAN);
  }

  ~ControlPoint()
  {
  }

  int m_id;
  double m_timestamp;
  std::string m_label;
  CONTROL_POINT_TYPE m_type;
  Eigen::Vector3d m_position;
  Eigen::Quaterniond m_orientation;
};

struct PointsPair
{
  Eigen::Vector3d xyz;
  Eigen::Vector3d cpt;
  double timestamp;
  int id;
  std::string name;
};

Eigen::Vector3d calculateAverageTranslation(const std::vector<Eigen::Vector3d>& translations, double& max_err)
{
  Eigen::Vector3d mean_xyz{0, 0, 0};
  for (const auto& t : translations)
  {
    mean_xyz += t;
  }
  mean_xyz = mean_xyz / translations.size();

  for (const auto& t : translations)
  {
    max_err = std::max((t - mean_xyz).norm(), max_err);
  }
  return mean_xyz;
}

double calculateAccumulateDistance(
    const std::vector<LidarFrame::Ptr>& lidar_frame_vec,
    const double start_time,
    const double end_time)
{
  double distance = 0;
  Eigen::Vector3d last_position;
  for (size_t i = 1; i < lidar_frame_vec.size() - 1; i++)
  {
    if (lidar_frame_vec[i]->m_timestamp > start_time && lidar_frame_vec[i]->m_timestamp < end_time)
    {
      distance += (lidar_frame_vec[i]->m_pose.translation() - last_position).norm();
    }
    if (lidar_frame_vec[i]->m_timestamp > end_time)
      break;

    last_position = lidar_frame_vec[i]->m_pose.translation();
  }

  return distance;
}

// 计算多个四元数的平均值
Eigen::Quaterniond calculateAverageQuatenion(const std::vector<Eigen::Quaterniond>& quaternions)
{
  Eigen::Quaterniond result(0.0, 0.0, 0.0, 0.0);
  for (const auto& q : quaternions)
  {
    result.coeffs() += q.coeffs();
  }
  result.normalize();
  return result;
}

int loadControlPoint(const std::string& filename, std::vector<ControlPoint::Ptr>& control_points_buf)
{
  std::fstream control_points_file;
  control_points_file.open(filename, std::fstream::in);
  if (!control_points_file.is_open())
  {
    return -1;
  }

  int n = 0;

  while (!control_points_file.eof())
  {
    char str[256];
    control_points_file.getline(str, sizeof(str));
    if (0 == strlen(str) || std::string(str).find("#") != std::string::npos)
      continue;

    double timestamp, x, y, z, qw, qx, qy, qz;
    int id;
    int type;
    char label[256];

    // timestamp id type label x y z qw qx qy qz
    sscanf(
        str,
        "%lf %d %d %s %lf %lf %lf %lf %lf %lf %lf",
        &timestamp,
        &id,
        &type,
        &label,
        &x,
        &y,
        &z,
        &qw,
        &qx,
        &qy,
        &qz);

    ControlPoint::Ptr control_point(new ControlPoint);
    control_point->m_timestamp = timestamp;
    control_point->m_id = id;
    control_point->m_label = std::string(label);
    control_point->m_type = (CONTROL_POINT_TYPE)type;
    control_point->m_position = Eigen::Vector3d(x, y, z);
    control_point->m_orientation = Eigen::Quaterniond(qw, qx, qy, qz);

    control_points_buf.push_back(control_point);
    n++;
  }

  printf("load control points: %lu\n", control_points_buf.size());
  control_points_file.close();

  return 0;
}

int loadLidarPoses(const std::string& filename, std::vector<LidarFrame::Ptr>& lidar_frame_buf)
{
  std::fstream pose_file;
  pose_file.open(filename, std::fstream::in);
  if (!pose_file.is_open())
  {
    return -1;
  }

  // camera poses:  timestamp, x, y, z, qw, qx, qy, qz, vx, vy, vz, grav_x, grav_y, grav_z
  int n = 0;
  while (!pose_file.eof())
  {
    char str[256];
    pose_file.getline(str, sizeof(str));
    if (0 == strlen(str))
      continue;

    double timestamp, x, y, z, qw, qx, qy, qz, vx, vy, vz, grav_x, grav_y, grav_z;
    sscanf(
        str,
        "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf",
        &timestamp,
        &x,
        &y,
        &z,
        &qw,
        &qx,
        &qy,
        &qz,
        &vx,
        &vy,
        &vz,
        &grav_x,
        &grav_y,
        &grav_z);

    LidarFrame::Ptr lidar_frame(new LidarFrame);
    lidar_frame->m_id = n++;
    lidar_frame->m_timestamp = timestamp;
    lidar_frame->m_pose = SE3(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(x, y, z));
    lidar_frame->m_pose_opt = lidar_frame->m_pose;
    lidar_frame->m_gravity_vec = Eigen::Vector3d(grav_x, grav_y, grav_z);

    lidar_frame_buf.push_back(lidar_frame);
  }

  printf("load poses: %lu\n", lidar_frame_buf.size());
  pose_file.close();
  return 0;
}

int loadFullLioStates(const std::string& pose_filename, std::vector<LidarFrame::Ptr>& lidar_frame_buf, bool print_flg)
{
  std::fstream pose_file;
  pose_file.open(pose_filename, std::ios::in | std::ios::binary);
  if (!pose_file.is_open())
  {
    printf("open full_states.bin failed:%s\n", pose_filename.c_str());
    return -1;
  }

  int n = 0;
  LioState lio_state;
  while (!pose_file.eof())
  {
    pose_file.read((char*)(&lio_state), sizeof(LioState));

    LidarFrame::Ptr lidar_frame(new LidarFrame);
    lidar_frame->m_id = n++;
    lidar_frame->m_timestamp = lio_state.timestamp;
    lidar_frame->m_pose = SE3(lio_state.q, lio_state.p);
    lidar_frame->m_pose_opt = lidar_frame->m_pose;
    lidar_frame->m_gravity_vec = lio_state.grav;

    if (print_flg)
    {
      printf(
          "%lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf %lf\n",
          lio_state.timestamp,
          lio_state.p(0),
          lio_state.p(1),
          lio_state.p(2),
          lio_state.q.w(),
          lio_state.q.x(),
          lio_state.q.y(),
          lio_state.q.z(),
          lio_state.v(0),
          lio_state.v(1),
          lio_state.v(2),
          lio_state.ba(0),
          lio_state.ba(1),
          lio_state.ba(2),
          lio_state.bg(0),
          lio_state.bg(1),
          lio_state.bg(2),
          lio_state.grav(0),
          lio_state.grav(1),
          lio_state.grav(2));
    }

    lidar_frame_buf.push_back(lidar_frame);
  }

  pose_file.close();

  double start_time = lidar_frame_buf.front()->m_timestamp;
  double end_time = lidar_frame_buf.back()->m_timestamp;
  return 0;
}

int main(int argc, char** argv)
{
  if (argc != 3)
  {
    std::cout << "invalid param, it should be:\ncheck_poses_by_control_point [pose file] [control "
                 "points file]"
              << std::endl;
    return -1;
  }

  std::string pose_filename(argv[1]);
  std::string control_points_filename(argv[2]);

  // Eigen::Vector3d t_imu_target{0.02243, -0.11746, -0.2094};
  Eigen::Vector3d t_imu_target{0.00882, -0.11555, -0.2093};

  std::vector<LidarFrame::Ptr> lidar_frame_vec;
  if (pose_filename.find(".csv") != std::string::npos)
    loadLidarPoses(pose_filename, lidar_frame_vec);
  else if (pose_filename.find(".bin") != std::string::npos)
    loadFullLioStates(pose_filename, lidar_frame_vec, true);

  std::vector<ControlPoint::Ptr> control_points_vec;
  loadControlPoint(control_points_filename, control_points_vec);

  std::sort(
      control_points_vec.begin(),
      control_points_vec.end(),
      [](const ControlPoint::Ptr& a, const ControlPoint::Ptr& b) { return a->m_timestamp < b->m_timestamp; });

  std::vector<PointsPair> points_pairs_vec;
  for (auto& cpt : control_points_vec)
  {
    // find lidar poses by timestamp
    std::vector<Eigen::Quaterniond> q_vec;
    std::vector<Eigen::Vector3d> t_vec;
    for (auto& lidar_frame : lidar_frame_vec)
    {
      if (fabs(lidar_frame->m_timestamp - cpt->m_timestamp) < 0.5)
      {
        q_vec.push_back(lidar_frame->m_pose.unit_quaternion());
        t_vec.push_back(lidar_frame->m_pose.translation());
      }

      if (lidar_frame->m_timestamp - cpt->m_timestamp > 2.0)
        break;
    }

    if (q_vec.empty())
      continue;

    double max_xyz_err = 0;
    Eigen::Vector3d mean_xyz = calculateAverageTranslation(t_vec, max_xyz_err);

    Eigen::Quaterniond mean_q = calculateAverageQuatenion(q_vec);
    // Eigen::Quaterniond mean_q = q_vec[q_vec.size() / 2];

    // transform imu pose to target pose
    Eigen::Vector3d target_xyz = mean_q * t_imu_target + mean_xyz;
    std::cout << std::fixed << cpt->m_timestamp << " " << cpt->m_label << " " << target_xyz.transpose() << " |"
              << max_xyz_err << " " << mean_xyz.transpose() << std::endl;

    PointsPair points_pair;
    points_pair.xyz = target_xyz;
    points_pair.cpt = cpt->m_position;
    points_pair.name = cpt->m_label;
    points_pair.timestamp = cpt->m_timestamp;
    points_pairs_vec.push_back(points_pair);
  }

  // compute diff
  int dim = points_pairs_vec.size();
  Eigen::MatrixXd diff_mat_groudtruth_xyz(dim, dim);
  Eigen::MatrixXd diff_mat_groudtruth_xy(dim, dim);
  Eigen::MatrixXd diff_mat_groudtruth_z(dim, dim);
  Eigen::MatrixXd diff_mat_xyz(dim, dim);
  Eigen::MatrixXd diff_mat_xy(dim, dim);
  Eigen::MatrixXd diff_mat_z(dim, dim);
  diff_mat_groudtruth_xyz.setZero();
  diff_mat_groudtruth_xy.setZero();
  diff_mat_groudtruth_z.setZero();
  diff_mat_xyz.setZero();
  diff_mat_xy.setZero();
  diff_mat_z.setZero();

  for (size_t r = 0; r < dim; r++)
  {
    for (size_t c = 0; c < dim; c++)
    {
      Eigen::Vector3d diff_gt_xyz = points_pairs_vec[r].cpt - points_pairs_vec[c].cpt;
      diff_mat_groudtruth_xyz(r, c) = diff_gt_xyz.norm();
      diff_mat_groudtruth_xy(r, c) = diff_gt_xyz.head(2).norm();
      diff_mat_groudtruth_z(r, c) = fabs(diff_gt_xyz(2));

      Eigen::Vector3d diff_xyz = points_pairs_vec[r].xyz - points_pairs_vec[c].xyz;
      diff_mat_xyz(r, c) = diff_xyz.norm();
      diff_mat_xy(r, c) = diff_xyz.head(2).norm();
      diff_mat_z(r, c) = fabs(diff_xyz(2));

      if (c < r)
      {
        diff_mat_groudtruth_xyz(r, c) = std::numeric_limits<double>::quiet_NaN();
        diff_mat_groudtruth_xy(r, c) = std::numeric_limits<double>::quiet_NaN();
        diff_mat_groudtruth_z(r, c) = std::numeric_limits<double>::quiet_NaN();
        diff_mat_xyz(r, c) = std::numeric_limits<double>::quiet_NaN();
        diff_mat_xy(r, c) = std::numeric_limits<double>::quiet_NaN();
        diff_mat_z(r, c) = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }

  Eigen::MatrixXd odom_err_rate_mat(dim, dim);
  odom_err_rate_mat.setZero();
  for (size_t r = 0; r < dim; r++)
  {
    for (size_t c = r; c < dim; c++)
    {
      double err = fabs(diff_mat_groudtruth_xyz(r, c) - diff_mat_xyz(r, c));
      double distance =
          calculateAccumulateDistance(lidar_frame_vec, points_pairs_vec[r].timestamp, points_pairs_vec[c].timestamp);
      odom_err_rate_mat(r, c) = err / distance * 100.;
    }
  }

  std::cout << std::setprecision(3) << "control points:\n" << diff_mat_groudtruth_xyz << std::endl;
  std::cout << std::setprecision(3) << "lio poses(target):\n" << diff_mat_xyz << std::endl;
  std::cout << std::setprecision(3) << "diff_xyz:\n" << (diff_mat_groudtruth_xyz - diff_mat_xyz) << std::endl;
  // std::cout << std::setprecision(3) << "diff_xy:\n"
  //           << (diff_mat_groudtruth_xy - diff_mat_xy) << std::endl;
  // std::cout << std::setprecision(3) << "diff_z:\n"
  //           << (diff_mat_groudtruth_z - diff_mat_z) << std::endl;
  std::cout << std::setprecision(3) << "acumulated odometry error rate(%):\n" << odom_err_rate_mat << std::endl;

  return 0;
}