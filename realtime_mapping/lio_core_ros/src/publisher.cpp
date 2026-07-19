#include <nav_msgs/Odometry.h>
#include <nav_msgs/Path.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#ifdef __linux__
#include <tf/LinearMath/Transform.h>
#include <tf/transform_broadcaster.h>
#endif

// clang-format off
#include <ros/message_event.h>
#include <pcl_conversions/pcl_conversions.h>
// clang-format on

#include "lio_core_ros/publisher.h"

namespace
{

void quaternionEigenToMsg(const Eigen::Quaterniond& q, geometry_msgs::Quaternion& msg)
{
  msg.x = q.x();
  msg.y = q.y();
  msg.z = q.z();
  msg.w = q.w();
}

void vectorEigenToMsg(const Eigen::Vector3d& v, geometry_msgs::Vector3& msg)
{
  msg.x = v.x();
  msg.y = v.y();
  msg.z = v.z();
}

}  // namespace

Publisher::Publisher(const FullParameters& params, const std::string& output_dir, const std::string& output_dir_temp)
    : params_(params), data_io_(output_dir_temp), output_dir_(output_dir), output_dir_temp_(output_dir_temp)
{
}

void Publisher::saveUlog(const lixel::LioResultMsg::Ptr& msg)
{
  if (!log_storage_)
  {
    log_storage_.reset(new middleware::ULogStorage());
    log_storage_->startLog(output_dir_ + "/lio.ulg");
    lslog(LSLOG_INFO) << "start ulog: " << output_dir_ + "/lio.ulg";
  }
  log_storage_->HandleFullState(msg->full_state);
  log_storage_->HandleIeskfAttribute(msg->attribute_ieskf);
  log_storage_->HandleIeskfStatePredict(msg->attribute_ieskf.sweep_id, msg->attribute_ieskf.state_predict);
  log_storage_->HandleIeskfAttributePredict(msg->attribute_ieskf.sweep_id, msg->attribute_ieskf.attritube_predict);
  log_storage_->HandleDebugMsgs(msg->debug_msgs);
}

void Publisher::saveXbc(const lixel::LioResultMsg::Ptr& msg)
{
  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(*msg->body_points, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(msg->full_state.timestamp);
  laserCloudmsg.header.frame_id = "body";
  data_io_.AddUndistortedLidarScan(laserCloudmsg);

  lixel_msgs::LioFullStates lio_state;
  lio_state.header.stamp = ros::Time().fromSec(msg->full_state.timestamp);
  quaternionEigenToMsg(msg->full_state.q, lio_state.q);
  vectorEigenToMsg(msg->full_state.p, lio_state.p);
  vectorEigenToMsg(msg->full_state.v, lio_state.v);
  vectorEigenToMsg(msg->full_state.ba, lio_state.ba);
  vectorEigenToMsg(msg->full_state.bg, lio_state.bg);
  vectorEigenToMsg(msg->full_state.gravity, lio_state.grav);
  data_io_.AddLocalLioStates(lio_state);

  for (auto& imu_item : msg->imu_vec)
  {
    static double last_imu_time = 0;
    if (imu_item.timestamp <= last_imu_time)
    {
      continue;
    }
    last_imu_time = imu_item.timestamp;

    sensor_msgs::Imu imu_msg;
    imu_msg.header.stamp.fromSec(imu_item.timestamp);
    imu_msg.header.frame_id = "body";
    vectorEigenToMsg(imu_item.acc, imu_msg.linear_acceleration);
    vectorEigenToMsg(imu_item.gyro, imu_msg.angular_velocity);
    data_io_.AddCorrectedImu(imu_msg);
  }
}

#ifdef SAVE_LAS
void Publisher::saveLas(const lixel::LioResultMsg::Ptr& msg)
{
  const pcl::PointCloud<pcl::PointXYZI> points_in;
  if (first_flag)
  {
    header.SetVersionMajor(1);
    header.SetVersionMinor(2);
    header.SetDataFormatId(liblas::ePointFormat3);
    header.SetScale(0.001, 0.001, 0.001);
    std::string file_name = output_dir_ + "/lio.las";
    liblas::Create(m_output_las, file_name);
    first_flag = false;
  }
  // write header
  liblas::Writer writer(m_output_las, header);
  header.SetMax(10000, 10000, 10000);
  header.SetMin(-10000, -10000, -10000);
  writer.SetHeader(header);
  writer.WriteHeader();

  // write points
  int counter = 0;
  Eigen::Quaterniond q(msg->full_state.q);
  Eigen::Matrix3d rot = q.toRotationMatrix();
  Eigen::Vector3d pos(msg->full_state.p);
  double timestamp = msg->full_state.timestamp;
  const auto& body_pcl = msg->body_points;
  for (size_t ii = 0; ii < body_pcl->size(); ii++)
  {
    const lixel::PointT pi = body_pcl->points[ii];
    Eigen::Vector3d input_points(pi.x, pi.y, pi.z);
    input_points = rot * input_points + pos;

    liblas::Point po(&header);
    po.SetCoordinates(input_points.x(), input_points.y(), input_points.z());
    po.SetIntensity((unsigned short)pi.intensity);
    po.SetTime(timestamp);
    writer.WritePoint(po);
    ++counter;
  }
  point_size += counter;
  header.SetPointRecordsCount(point_size);
  m_output_las.flush();
}
#endif

void Publisher::publishToRviz(const lixel::LioResultMsg::Ptr& msg)
{
#ifdef __linux__
  lixel::PointCloud laserCloudWorld = *msg->body_points;
  lixel::FullStateMsg state = msg->full_state;
  laserCloudWorld.sensor_origin_ = Eigen::Vector4f(state.p[0], state.p[1], state.p[2], 0);
  laserCloudWorld.sensor_orientation_ = Eigen::Quaternionf(state.q);

  static tf::TransformBroadcaster br;
  tf::Transform transform;
  tf::Quaternion q;
  transform.setOrigin(tf::Vector3(state.p[0], state.p[1], state.p[2]));
  q.setW(state.q.w());
  q.setX(state.q.x());
  q.setY(state.q.y());
  q.setZ(state.q.z());
  transform.setRotation(q);
  br.sendTransform(tf::StampedTransform(transform, ros::Time().fromSec(state.timestamp), "world", "body"));

  sensor_msgs::PointCloud2 laserCloudmsg;
  pcl::toROSMsg(laserCloudWorld, laserCloudmsg);
  laserCloudmsg.header.stamp = ros::Time().fromSec(state.timestamp);
  laserCloudmsg.header.frame_id = "body";
  pub_laser_cloud_full_.publish(laserCloudmsg);

  static nav_msgs::Path m_path_for_pub;
  geometry_msgs::PoseStamped msg_body_pose;
  msg_body_pose.pose.position.x = state.p[0];
  msg_body_pose.pose.position.y = state.p[1];
  msg_body_pose.pose.position.z = state.p[2];
  msg_body_pose.pose.orientation.x = state.q.x();
  msg_body_pose.pose.orientation.y = state.q.y();
  msg_body_pose.pose.orientation.z = state.q.z();
  msg_body_pose.pose.orientation.w = state.q.w();
  msg_body_pose.header.stamp = ros::Time().fromSec(state.timestamp);
  msg_body_pose.header.frame_id = "world";
  m_path_for_pub.header.frame_id = "world";
  m_path_for_pub.poses.push_back(msg_body_pose);
  pub_path_.publish(m_path_for_pub);

  nav_msgs::Odometry odomAftMapped;
  odomAftMapped.header.frame_id = "world";
  odomAftMapped.child_frame_id = "body";
  odomAftMapped.header.stamp = msg_body_pose.header.stamp;
  odomAftMapped.pose.pose.position.x = state.p[0];
  odomAftMapped.pose.pose.position.y = state.p[1];
  odomAftMapped.pose.pose.position.z = state.p[2];
  odomAftMapped.pose.pose.orientation.x = state.q.x();
  odomAftMapped.pose.pose.orientation.y = state.q.y();
  odomAftMapped.pose.pose.orientation.z = state.q.z();
  odomAftMapped.pose.pose.orientation.w = state.q.w();
  pub_odom_after_mapped_.publish(odomAftMapped);

  for (auto& imu_item : msg->imu_vec)
  {
    static double last_imu_time = 0;
    if (imu_item.timestamp <= last_imu_time)
    {
      continue;
    }
    last_imu_time = imu_item.timestamp;

    sensor_msgs::Imu imu_msg;
    imu_msg.header.stamp.fromSec(imu_item.timestamp);
    imu_msg.header.frame_id = "body";
    vectorEigenToMsg(imu_item.acc, imu_msg.linear_acceleration);
    vectorEigenToMsg(imu_item.gyro, imu_msg.angular_velocity);
    pub_imu_.publish(imu_msg);
  }
#endif
}

void Publisher::publishLioResultAsCallback(const lixel::LioResultMsg::Ptr& msg)
{
  std::lock_guard<std::mutex> lg(mtx_for_exit_);
  if (exit_)
  {
    return;
  }

  publishToRviz(msg);

  saveUlog(msg);

  saveXbc(msg);

#ifdef SAVE_LAS
  saveLas(msg);
#endif
}

void Publisher::registerCallbacks(std::shared_ptr<lixel::LioCore> lio_core)
{
  lio_core->setLioResultCallback(std::bind(&Publisher::publishLioResultAsCallback, this, std::placeholders::_1));
}

Publisher::~Publisher()
{
  stop();
}

void Publisher::stop()
{
  std::lock_guard<std::mutex> lg(mtx_for_exit_);
  if (exit_)
  {
    return;
  }
  lslog(LSLOG_INFO) << "exiting publisher";
  data_io_.Close();
  if (log_storage_)
  {
    lslog(LSLOG_INFO) << "stop ulog: " << output_dir_ + "/lio.ulg";
    log_storage_->stopLog();
  }
  exit_ = true;
}

#ifdef __linux__
void Publisher::setNodeHandle(ros::NodeHandle& nh)
{
  nh_ = nh;
  pub_laser_cloud_full_ = nh_.advertise<sensor_msgs::PointCloud2>("/cloud_registered", 5);
  pub_path_ = nh_.advertise<nav_msgs::Path>("/path", 5);
  pub_odom_after_mapped_ = nh_.advertise<nav_msgs::Odometry>("/odometry", 5);
  pub_imu_ = nh_.advertise<sensor_msgs::Imu>("/imu_corrected", 5);
}
#endif
