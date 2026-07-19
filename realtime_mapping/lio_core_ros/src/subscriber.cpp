#include <rosbag/view.h>
#ifdef __linux__
#include <boost/scope_exit.hpp>
#endif

// clang-format off
#include <ros/message_event.h>
#include <pcl_conversions/pcl_conversions.h>
// clang-format on

#include "lio_core_ros/subscriber.h"
#include "lixel_msgs/CustomMsg.h"
#include "log/lsLogger.h"

namespace hesai_ros
{
struct EIGEN_ALIGN16 Point
{
  PCL_ADD_POINT4D
  float intensity;
  double timestamp;
  uint16_t ring;
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW
};
}  // namespace hesai_ros
POINT_CLOUD_REGISTER_POINT_STRUCT(
    hesai_ros::Point,
    (float,
     x,
     x)(float, y, y)(float, z, z)(float, intensity, intensity)(double, timestamp, timestamp)(uint16_t, ring, ring))

namespace
{

lixel::ImuMsg::Ptr fromROS(const sensor_msgs::ImuConstPtr& msg)
{
  lixel::ImuMsg::Ptr msg_out{new lixel::ImuMsg()};
  msg_out->timestamp = msg->header.stamp.toSec();
  msg_out->gyro[0] = msg->angular_velocity.x;
  msg_out->gyro[1] = msg->angular_velocity.y;
  msg_out->gyro[2] = msg->angular_velocity.z;
  msg_out->acc[0] = msg->linear_acceleration.x;  // here the source acc unit is m/s^2, not g
  msg_out->acc[1] = msg->linear_acceleration.y;
  msg_out->acc[2] = msg->linear_acceleration.z;
  return msg_out;
}

lixel::MotorMsg::Ptr fromROS(const nav_msgs::OdometryConstPtr& msg)
{
  lixel::MotorMsg::Ptr msg_out{new lixel::MotorMsg()};
  msg_out->timestamp = msg->header.stamp.toSec();
  msg_out->q.x() = msg->pose.pose.orientation.x;
  msg_out->q.y() = msg->pose.pose.orientation.y;
  msg_out->q.z() = msg->pose.pose.orientation.z;
  msg_out->q.w() = msg->pose.pose.orientation.w;
  return msg_out;
}

lixel::GnssMsg::Ptr fromROS(const sensor_msgs::NavSatFixConstPtr& msg)
{
  // todo kk fill fields
  lixel::GnssMsg::Ptr msg_out{new lixel::GnssMsg(
      msg->header.stamp.toSec(),
      msg->status.status,
      lixel::V3D{msg->longitude, msg->latitude, msg->altitude},
      lixel::V3D(msg->position_covariance[0], msg->position_covariance[4], msg->position_covariance[8]),
      0,
      0,
      0,
      lixel::GcsType::WGS_84)};
  return msg_out;
}

lixel::PointCloudMsg::Ptr fromROS(const sensor_msgs::PointCloud2ConstPtr& msg, int frame_id)
{
  lixel::PointCloudMsg::Ptr msg_out{new lixel::PointCloudMsg()};
  msg_out->frame_id = frame_id;
  msg_out->points.reset(new lixel::PointCloud);
  msg_out->timestamp = msg->header.stamp.toSec();
  pcl::PointCloud<hesai_ros::Point> pl_orig;
  pcl::fromROSMsg(*msg, pl_orig);
  for (auto& pt : pl_orig)
  {
    lixel::PointT pt_new;
    pt_new.x = pt.x;
    pt_new.y = pt.y;
    pt_new.z = pt.z;
    pt_new.intensity = pt.intensity;
    pt_new.ring = pt.ring;
    pt_new.timestamp = pt.timestamp;
    msg_out->points->push_back(pt_new);
  }
  return msg_out;
}

lixel::PointCloudMsg::Ptr fromROS(const lixel_msgs::CustomMsgConstPtr& msg, int frame_id)
{
  lixel::PointCloudMsg::Ptr msg_out{new lixel::PointCloudMsg()};
  msg_out->frame_id = frame_id;
  msg_out->points.reset(new lixel::PointCloud);
  msg_out->timestamp = msg->header.stamp.toSec();
  for (auto& pt : msg->points)
  {
    lixel::PointT pt_new;
    pt_new.x = pt.x;
    pt_new.y = pt.y;
    pt_new.z = pt.z;
    pt_new.intensity = pt.reflectivity;
    pt_new.ring = pt.line;
    pt_new.timestamp = pt.offset_time * 1e-9 + msg_out->timestamp;
    msg_out->points->push_back(pt_new);
  }
  return msg_out;
}

class RateCounter
{
 public:
  RateCounter() = default;

  void addDataSample(double timestamp)
  {
    if (sample_count_ == 0)
    {
      start_time_ = timestamp;
    }
    ++sample_count_;
    end_time_ = timestamp;
  }

  double getRate()
  {
    if (sample_count_ <= 1)
    {
      return -1;  // return -1 means message will be played at the highest rate
    }
    return sample_count_ / (end_time_ - start_time_);
  }

  void sleepBySensorRateAndSpeedRatio(double speed_ratio)
  {
    double rate = getRate();
    if (rate < 0)
    {
      return;
    }
    std::this_thread::sleep_for(std::chrono::duration<double>(1.0 / (speed_ratio * rate)));
  }

  double getDuration()
  {
    if (sample_count_ <= 1)
    {
      return 0;
    }
    return end_time_ - start_time_;
  }

 private:
  double start_time_ = -1;  // -1 means default value
  double end_time_;
  int sample_count_ = 0;
};

}  // namespace

Subscriber::Subscriber(const FullParameters& params, CalibModel calib_model)
    : params_(params), calib_model_(calib_model)
{
}

void Subscriber::registerHandlers(const std::shared_ptr<lixel::LioCore>& lio_core)
{
#ifdef __linux__

  lio_core_ = lio_core;

#ifdef __x86_64__
  bool on_x86 = true;
#else
  bool on_x86 = false;
#endif

  // use different queue size for different platforms
  sub_imu_ = nh_.subscribe(
      params_.ros_param.imu_topic, on_x86 ? INT_MAX : ARM_IMU_QUEUE_SIZE, &Subscriber::imuMsgCallback, this);
  sub_encoder_ = nh_.subscribe(
      params_.ros_param.encoder_topic,
      on_x86 ? INT_MAX : ARM_ENCODER_QUEUE_SIZE,
      &Subscriber::encoderMsgCallback,
      this);
  if (calib_model_ == CalibModel::L2_LIKE)
  {
    sub_lidar_ = nh_.subscribe<sensor_msgs::PointCloud2>(
        params_.ros_param.lidar_topic,
        on_x86 ? INT_MAX : ARM_LIDAR_QUEUE_SIZE,
        [this](const sensor_msgs::PointCloud2ConstPtr& raw_msg) { this->lidarMsgCallback(raw_msg, 0); });
  }
  else if (calib_model_ == CalibModel::K1)
  {
    sub_lidar_ = nh_.subscribe<lixel_msgs::CustomMsg>(
        params_.ros_param.lidar_topic,
        on_x86 ? INT_MAX : ARM_LIDAR_QUEUE_SIZE,
        [this](const lixel_msgs::CustomMsgConstPtr& raw_msg) { this->lidarMsgCallback(raw_msg, 0); });
  }
  else
  {
    lslog(LSLOG_ERROR) << "Unsupported device model: " << static_cast<int>(calib_model_);
  }
  sub_gnss_ = nh_.subscribe(
      params_.ros_param.gnss_topic, on_x86 ? INT_MAX : ARM_GNSS_QUEUE_SIZE, &Subscriber::gnssMsgCallback, this);

#endif
}

void Subscriber::imuMsgCallback(const sensor_msgs::ImuConstPtr& msg)
{
  static sensor_msgs::ImuConstPtr last_msg;

  if (!last_msg)
  {
    lio_core_->addSensorData(fromROS(msg));
    last_msg = msg;
    return;
  }

  double last_time = last_msg->header.stamp.toSec();
  double cur_time = msg->header.stamp.toSec();

  // if a msg is out of order, just ignore it
  if (cur_time < last_time)
  {
    lslog(LSLOG_WARNING) << "imu data out of order, skipped.";
    return;
  }

  if (cur_time - last_time > TIME_DATA_GAP_IMU)
  {
    lslog(LSLOG_ERROR) << "Fatal imu data missing.";
  }

  std::shared_ptr<void> sg1(nullptr, [msg](void*) { last_msg = msg; });

  if (cur_time == last_time)
  {
    lslog(LSLOG_WARNING) << "Reapted imu data.";
    return;
  }

  if (cur_time - last_time > 0.01)
  {
    lslog(LSLOG_WARNING) << "Tolerable imu data missing.";
  }

  lio_core_->addSensorData(fromROS(msg));
}

void Subscriber::encoderMsgCallback(const nav_msgs::OdometryConstPtr& msg)
{
  static nav_msgs::OdometryConstPtr last_msg;

  if (!last_msg)
  {
    lio_core_->addSensorData(fromROS(msg));
    last_msg = msg;
    return;
  }

  double last_time = last_msg->header.stamp.toSec();
  double cur_time = msg->header.stamp.toSec();

  // if a msg is out of order, just ignore it
  if (cur_time < last_time)
  {
    lslog(LSLOG_WARNING) << "encoder data out of order, skipped.";
    return;
  }

  if (cur_time - last_time > TIME_DATA_GAP_ENCODER)
  {
    lslog(LSLOG_ERROR) << "Fatal encoder data missing.";
  }

  std::shared_ptr<void> sg1(nullptr, [msg](void*) { last_msg = msg; });

  if (cur_time == last_time)
  {
    lslog(LSLOG_WARNING) << "Reapted encoder data.";
    return;
  }

  if (cur_time - last_time > 0.01)
  {
    lslog(LSLOG_WARNING) << "Tolerable encoder data missing.";
  }

  lio_core_->addSensorData(fromROS(msg));
}

void Subscriber::lidarMsgCallback(const sensor_msgs::PointCloud2ConstPtr& raw_msg, int frame_id)
{
  this->lidarMsgCallback(fromROS(raw_msg, frame_id));
}

void Subscriber::lidarMsgCallback(const lixel_msgs::CustomMsgConstPtr& msg, int frame_id)
{
  this->lidarMsgCallback(fromROS(msg, frame_id));
}

void Subscriber::gnssMsgCallback(const sensor_msgs::NavSatFixConstPtr& msg)
{
  // todo kk check exception
  lio_core_->addSensorData(fromROS(msg));
}

void Subscriber::setLioCore(const std::shared_ptr<lixel::LioCore>& lio_core)
{
  lio_core_ = lio_core;
}

LioCoreRosErrorCode Subscriber::processBag(const rosbag::Bag& bag, volatile int& stop_signal_recv, double* progress)
{
  // count rate
  RateCounter lidar_rate;
  RateCounter imu_rate;
  RateCounter encoder_rate;
  int num_of_lidar = 0;
  for (auto& m : rosbag::View(bag))
  {
    if (stop_signal_recv)
    {
      return LioCoreRosErrorCode::EC_LIO_OK;
    }
    if ((m.isType<sensor_msgs::PointCloud2>() || m.isType<lixel_msgs::CustomMsg>()) &&
        (m.getTopic() == params_.ros_param.lidar_topic))
    {
      lidar_rate.addDataSample(m.getTime().toSec());
      ++num_of_lidar;
    }
    else if (m.isType<sensor_msgs::Imu>() && (m.getTopic() == params_.ros_param.imu_topic))
    {
      imu_rate.addDataSample(m.getTime().toSec());
    }
    else if (m.isType<nav_msgs::Odometry>() && (m.getTopic() == params_.ros_param.encoder_topic))
    {
      encoder_rate.addDataSample(m.getTime().toSec());
    }
  }
  lslog(LSLOG_INFO) << "lidar rate: " << lidar_rate.getRate();
  lslog(LSLOG_INFO) << "imu rate: " << imu_rate.getRate();
  lslog(LSLOG_INFO) << "encoder rate: " << encoder_rate.getRate();

  // return error code if data is not enough
  if (lidar_rate.getDuration() < TIME_DATA_NOT_ENOUGH_THRESHOLD)
  {
    lslog(LSLOG_ERROR) << "lidar data too short: " << lidar_rate.getDuration();
    return LioCoreRosErrorCode::EC_LIO_HBC_READ_LIDAR_FAILED;
  }
  if (imu_rate.getDuration() < TIME_DATA_NOT_ENOUGH_THRESHOLD)
  {
    lslog(LSLOG_ERROR) << "imu data too short: " << imu_rate.getDuration();
    return LioCoreRosErrorCode::EC_LIO_HBC_READ_IMU_FAILED;
  }
  if (params_.extrinsic_param.motor_param.enabled && encoder_rate.getDuration() < TIME_DATA_NOT_ENOUGH_THRESHOLD)
  {
    lslog(LSLOG_ERROR) << "encoder data too short: " << encoder_rate.getDuration();
    return LioCoreRosErrorCode::EC_LIO_HBC_READ_ENCODER_FAILED;
  }

  int frame_id = -1;
  // read and handle messages
  for (auto& m : rosbag::View(bag))
  {
    if ((m.isType<sensor_msgs::PointCloud2>() || m.isType<lixel_msgs::CustomMsg>()) &&
        (m.getTopic() == params_.ros_param.lidar_topic))
    {
      ++frame_id;
    }

    if (frame_id < params_.ros_param.offline_mode_start_frame_id)
    {
      continue;
    }
    if (frame_id >= params_.ros_param.offline_mode_end_frame_id || stop_signal_recv)
    {
      break;
    }

    if ((m.isType<sensor_msgs::PointCloud2>() || m.isType<lixel_msgs::CustomMsg>()) &&
        (m.getTopic() == params_.ros_param.lidar_topic))
    {
      if (m.isType<sensor_msgs::PointCloud2>())
      {
        lidarMsgCallback(m.instantiate<sensor_msgs::PointCloud2>(), frame_id);
      }
      else
      {
        lidarMsgCallback(m.instantiate<lixel_msgs::CustomMsg>(), frame_id);
      }
      lidar_rate.sleepBySensorRateAndSpeedRatio(params_.ros_param.offline_mode_speed_ratio);
    }
    else if (m.isType<sensor_msgs::Imu>() && (m.getTopic() == params_.ros_param.imu_topic))
    {
      imuMsgCallback(m.instantiate<sensor_msgs::Imu>());
    }
    else if (m.isType<nav_msgs::Odometry>() && (m.getTopic() == params_.ros_param.encoder_topic))
    {
      encoderMsgCallback(m.instantiate<nav_msgs::Odometry>());
    }

    if (progress)
    {
      *progress = std::min(1.0, (double)frame_id / num_of_lidar) * 100;
    }
  }

  return LioCoreRosErrorCode::EC_LIO_OK;
}

void Subscriber::lidarMsgCallback(const lixel::PointCloudMsg::Ptr& msg)
{
  static lixel::PointCloudMsg::Ptr last_msg;

  lslog(LSLOG_INFO) << "received lidar points: " << msg->points->size() << " @" << std::fixed << std::setprecision(3)
                    << msg->timestamp;
  if (!last_msg)
  {
    lio_core_->addSensorData(msg);
    last_msg = msg;
    return;
  }

  double last_time = last_msg->timestamp;
  double cur_time = msg->timestamp;

  // if a msg is out of order, just ignore it
  if (cur_time < last_time)
  {
    lslog(LSLOG_WARNING) << "lidar data out of order, skipped.";
    return;
  }

  if (cur_time - last_time > TIME_DATA_GAP_LIDAR)
  {
    lslog(LSLOG_ERROR) << "Fatal lidar data missing." << std::fixed << std::setprecision(6) << cur_time << " "
                       << last_time;
  }

  if (cur_time == last_time)
  {
    lslog(LSLOG_WARNING) << "Reapted lidar data.";
    return;
  }

  if (cur_time - last_time < 0.02)
  {
    lslog(LSLOG_WARNING) << "Lidar duration too short: " << cur_time - last_time << ", skipped.";
    return;
  }

  if (msg->points->size() < 3000)
  {
    lslog(LSLOG_WARNING) << "Lidar points too few: " << msg->points->size() << ", skipped.";
    return;
  }

#ifndef __x86_64__
  auto cur_timestamp = ros::Time();
  if (cur_timestamp.toSec() - cur_time > 30)
  {
    lslog(LSLOG_INFO) << "Lidar timestamp abnormal compared with ros time: " << cur_timestamp.toSec() - cur_time;
    return;
  }
#endif

  if (fabs(cur_time - msg->points->points.front().timestamp) > 30)
  {
    lslog(LSLOG_INFO) << std::fixed << std::setprecision(6) << "Lidar timestamp abnormal: header timestamp " << cur_time
                      << ", timestamp of first point " << msg->points->points.front().timestamp;
    return;
  }

  while (lio_core_->getInputDataCommonDuration() >
         MAX_CACHED_QUEUE_DURATION)  // block until data accumulation time enough less
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
  }
  lio_core_->addSensorData(msg);
  last_msg = msg;
}

#ifdef __linux__
void Subscriber::setNodeHandle(ros::NodeHandle& nh)
{
  nh_ = nh;
}
#endif
