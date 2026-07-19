#include <Eigen/Core>
#include <Eigen/Dense>
#include <opencv2/core/eigen.hpp>

#include <pcl/common/common.h>
#include <pcl/common/transforms.h>
#include <pcl/features/don.h>
#include <pcl/features/normal_3d.h>
#include <pcl/features/normal_3d_omp.h>  //使用OMP需要添加的头文件
#include <pcl/filters/conditional_removal.h>
#include <pcl/filters/crop_box.h>
#include <pcl/filters/normal_refinement.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <pcl/search/organized.h>
#include <pcl/segmentation/extract_clusters.h>
#include <pcl/surface/mls.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/range_image_visualizer.h>
#include <pcl_conversions/pcl_conversions.h>
#include "pcl/features/range_image_border_extractor.h"
#include "pcl/range_image/range_image.h"

#include <ros/ros.h>
#include <visualization_msgs/MarkerArray.h>
#ifdef __x86_64__
#include <rviz_visual_tools/rviz_visual_tools.h>
#endif
#include <nav_msgs/Odometry.h>
#include <sensor_msgs/PointCloud2.h>
#include <thread>

#include "voxel_map.hpp"

xvmp::VoxelMap voxelmap;
Eigen::Vector3d view_point;
ros::Publisher pub_map, pub_effect_points;
std::vector<xvmp::Plane::Ptr> all_planes;

static inline bool isRotationMatrix(const Eigen::Matrix3d &rot)
{
  Eigen::Vector3d evec0 = rot.block<3, 1>(0, 0);
  Eigen::Vector3d evec1 = rot.block<3, 1>(0, 1);
  Eigen::Vector3d evec2 = rot.block<3, 1>(0, 2);

  double d0 = (evec0.cross(evec1) - evec2).norm();
  double d1 = (evec2.cross(evec0) - evec1).norm();
  double d2 = (evec1.cross(evec2) - evec0).norm();

  return (d0 < 1e-5 && d1 < 1e-5 && d2 < 1e-5);
}

void publishPlanes(const ros::Publisher &pub_map, const std::vector<xvmp::Plane::Ptr> &planes)
{
#ifdef __x86_64__
  // rviz不会清空历史marker，手动清空
  static rviz_visual_tools::RvizVisualTools rviz_tool("world", "/planes");

  // id不要累加，rviz非常非常非常卡
  visualization_msgs::MarkerArray plane_makers;
  std::vector<visualization_msgs::Marker>().swap(plane_makers.markers);
  // plane_makers.markers.resize(0);

  for (size_t i = 0; i < planes.size(); i++)
  {
    xvmp::Plane::Ptr plane = planes[i];
    // if (!plane->is_plane || sqrt(plane->min_eigen_value) > 0.01)
    if (!plane->is_plane)
      continue;

    Eigen::Matrix3d rot = plane->eigen_vectors;
    Eigen::Vector3d evec0 = plane->eigen_vectors.block<3, 1>(0, 2);
    Eigen::Vector3d evec1 = plane->eigen_vectors.block<3, 1>(0, 1);
    Eigen::Vector3d evec2 = plane->eigen_vectors.block<3, 1>(0, 0);
    rot.block<3, 1>(0, 0) = evec0;
    rot.block<3, 1>(0, 1) = evec1;
    rot.block<3, 1>(0, 2) = evec2;
    if (!isRotationMatrix(rot))
      rot = -rot;
    Eigen::Quaterniond q(rot);

    if (plane->normal.norm() < 0.5 || plane->normal.norm() > 1.1)
    {
      std::cout << "normal:" << plane->normal.norm() << std::endl;
      std::cout << "eigen_values:"
                << Eigen::Vector3d(
                       plane->min_eigen_value, plane->mid_eigen_value, plane->max_eigen_value)
                       .transpose()
                << std::endl;
      std::cout << "eigen_vectors:\n" << plane->eigen_vectors << std::endl;
    }

    // plane
    visualization_msgs::Marker marker_plane;
    marker_plane.header.frame_id = "world";
    marker_plane.header.stamp = ros::Time();
    marker_plane.ns = std::to_string(plane->layer);

    marker_plane.id = i;
    marker_plane.type = visualization_msgs::Marker::CYLINDER;
    marker_plane.action = visualization_msgs::Marker::ADD;
    marker_plane.pose.position.x = plane->center[0];
    marker_plane.pose.position.y = plane->center[1];
    marker_plane.pose.position.z = plane->center[2];
    marker_plane.pose.orientation.w = q.w();
    marker_plane.pose.orientation.x = q.x();
    marker_plane.pose.orientation.y = q.y();
    marker_plane.pose.orientation.z = q.z();
    marker_plane.scale.x = 3 * sqrt(plane->max_eigen_value);
    marker_plane.scale.y = 3 * sqrt(plane->mid_eigen_value);
    marker_plane.scale.z = 3 * sqrt(plane->min_eigen_value);
    marker_plane.color.a = plane->planarity + 0.1;
    marker_plane.color.r = fabs(plane->normal(0));
    marker_plane.color.g = fabs(plane->normal(1));
    marker_plane.color.b = fabs(plane->normal(2));

    plane_makers.markers.push_back(marker_plane);

    // normal
    visualization_msgs::Marker marker_normal;
    marker_normal.header = marker_plane.header;
    marker_normal.ns = "normal";
    marker_normal.action = visualization_msgs::Marker::ADD;
    marker_normal.pose.orientation.w = 1.0;
    marker_normal.id = i;
    marker_normal.type = visualization_msgs::Marker::ARROW;
    marker_normal.scale.x = 0.01;  // 柄直径
    marker_normal.scale.y = 0.02;  // 箭头直径
    marker_normal.scale.z = 0.0;
    marker_normal.color.b = 1.0;
    marker_normal.color.a = 1.0;

    Eigen::Vector3d ps(plane->center(0), plane->center(1), plane->center(2));
    Eigen::Vector3d pe = ps + 0.4 * plane->normal;
    geometry_msgs::Point p;
    p.x = ps(0);
    p.y = ps(1);
    p.z = ps(2);
    marker_normal.points.push_back(p);  // The line list needs two points for each line

    p.x = pe(0);
    p.y = pe(1);
    p.z = pe(2);
    marker_normal.points.push_back(p);

    plane_makers.markers.push_back(marker_normal);
  }

  // pub_map.publish(plane_makers);
  rviz_tool.deleteAllMarkers();
  rviz_tool.setLifetime(0.001);
  rviz_tool.publishMarkers(plane_makers);
#endif
}

void publishPoints(const ros::Publisher &pub_points, const PointCloud::Ptr &points)
{
  sensor_msgs::PointCloud2 cloud_msg;
  pcl::toROSMsg(*points, cloud_msg);
  cloud_msg.header.stamp = ros::Time().fromSec(points->header.stamp / 1e6);
  cloud_msg.header.frame_id = "world";
  pub_points.publish(cloud_msg);
}

void pointsCallback(const sensor_msgs::PointCloud2::ConstPtr &msg)
{
  PointCloud::Ptr pcl_msg(new PointCloud);
  pcl::fromROSMsg(*msg, *pcl_msg);
  ROS_INFO("\nreceived points %lu", pcl_msg->points.size());

  double downsample_size = 0.2;

  PointCloud::Ptr points_filtered(new PointCloud);
  *points_filtered = *pcl_msg;

  // pcl::VoxelGrid<pcl::PointXYZINormal> ds_filter;
  // ds_filter.setLeafSize(downsample_size, downsample_size, downsample_size);
  // ds_filter.setInputCloud(pcl_msg);
  // ds_filter.filter(*points_filtered);
  // ROS_INFO("downsampled points %lu", points_filtered->points.size());

  int max_layer = 3;
  float voxel_size = downsample_size * pow(2, double(max_layer));
  std::vector<int> layer_point_size = {24, 12, 6, 5, 5};
  int max_points_size = 50;
  int max_cov_points_size = 50;
  float planar_threshold = 0.1;

  static int frame_count = 0;
  if (0 == frame_count++)
  {
    ROS_INFO("init voxel map");
    voxelmap.init(
        *points_filtered,
        voxel_size,
        max_layer,
        layer_point_size,
        max_points_size,
        max_cov_points_size,
        planar_threshold,
        view_point);
    return;
  }

  // voxelmap.get_searched_planes(searched_planes);
  double t0 = ros::Time::now().toSec();
  std::vector<xvmp::Plane::Ptr> searched_planes;
  PointCloud::Ptr searched_points(new PointCloud);
  for (size_t i = 0; i < points_filtered->size(); i++)
  {
    Eigen::Vector3d point(
        points_filtered->points[i].x, points_filtered->points[i].y, points_filtered->points[i].z);
    xvmp::Plane::Ptr plane = NULL;
    if (voxelmap.search_plane(point, view_point, plane, true, 0.05))
    {
      searched_planes.push_back(plane);
      searched_points->push_back(points_filtered->points[i]);
    }
  }
  ROS_INFO(
      "searched plane size: %lu, cost %.3lfs",
      searched_planes.size(),
      ros::Time::now().toSec() - t0);

  // update
  t0 = ros::Time::now().toSec();
  voxelmap.update(*points_filtered, view_point);
  ROS_INFO(
      "update voxel map  %lu,  cost %.3lfs",
      points_filtered->points.size(),
      ros::Time::now().toSec() - t0);

  publishPlanes(pub_map, searched_planes);
  publishPoints(pub_effect_points, searched_points);
}

void command()
{
  char buff[128] = {0};

  while (ros::ok())
  {
    if (NULL != fgets(buff, sizeof(buff), stdin))
    {
      char c;
      double value = 0.6;
      sscanf(buff, "%c %lf", &c, &value);

      if (c == 's')
      {
        voxelmap.get_all_planes(all_planes);
        ROS_INFO("all plane size: %lu", all_planes.size());

        ROS_INFO("publish voxel map...");
        publishPlanes(pub_map, all_planes);
      }
    }
  }
}

void odometryCallback(const nav_msgs::Odometry::ConstPtr &msg)
{
  view_point = Eigen::Vector3d(
      msg->pose.pose.position.x, msg->pose.pose.position.y, msg->pose.pose.position.z);
}

int main(int argc, char **argv)
{
  ros::init(argc, argv, "voxelmap_test");
  ros::console::set_logger_level(ROSCONSOLE_DEFAULT_NAME, ros::console::levels::Info);
  ros::NodeHandle nh;

  pub_map = nh.advertise<visualization_msgs::MarkerArray>("/planes", 100);
  pub_effect_points = nh.advertise<sensor_msgs::PointCloud2>("/points", 100);
  ros::Subscriber points_sub = nh.subscribe("/cloud_registered_ds", 100, &pointsCallback);
  ros::Subscriber odom_sub = nh.subscribe("/odometry", 100, &odometryCallback);

  std::thread keyboard_command_process = std::thread(command);

  ros::spin();
}