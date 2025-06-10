
#include <glog/logging.h>
#include <iomanip>

#include "msg_pack_synchronizer.h"

namespace {

void CollectLidarData(std::shared_ptr<PointCloud>& out_cloud, std::deque<PointXYZIRT>& buffer, double start_time, double end_time) {
  out_cloud.reset(new PointCloud());
  while (!buffer.empty() && buffer.front().timestamp < end_time) {
    out_cloud->push_back(buffer.front());
    buffer.pop_front();
  }
}

template <typename T>
void CollectImuOrEncoderData(std::vector<T>& out_msgs, std::deque<T>& buffer, double start_time, double end_time) {
  int begin_idx = std::distance(buffer.begin(), std::upper_bound(buffer.begin(), buffer.end(), start_time,
                                                                 [](double t, const T& msg) { return t < msg.timestamp; })) -
                  1;
  int end_idx = std::distance(buffer.begin(),
                              std::upper_bound(buffer.begin(), buffer.end(), end_time, [](double t, const T& msg) { return t < msg.timestamp; })) -
                1;
  CHECK_GE(begin_idx, 0);
  CHECK_LE(begin_idx, static_cast<int>(buffer.size()) - 2);
  CHECK_GE(end_idx, 0);
  CHECK_LE(end_idx, static_cast<int>(buffer.size()) - 2);
  for (int i = begin_idx; i <= end_idx + 1; ++i) {
    out_msgs.push_back(buffer[i]);
  }
}

}  // namespace

void MsgPackSynchronizer::AddLidarData(const LidarMsg::ConstPtr& lidar_msg) {
  for (const auto& lidar_point : *lidar_msg->lidar_points) {
    if (!lidar_data_queue_.empty() && lidar_point.timestamp <= lidar_data_queue_.back().timestamp) {
      DLOG(WARNING) << "sensor timestamp is not increasing";
      return;
    }
    if (!lidar_data_queue_.empty() && lidar_point.timestamp - lidar_data_queue_.back().timestamp > kMaxSensorTimestampGap) {
      DLOG(FATAL) << "Lidar timestamp gap is too large: " << lidar_point.timestamp - lidar_data_queue_.back().timestamp;
    }
    lidar_data_queue_.push_back(lidar_point);
  }
}

void MsgPackSynchronizer::AddImuData(const ImuMsg& imu_msg) {
  if (!imu_data_queue_.empty() && imu_msg.timestamp <= imu_data_queue_.back().timestamp) {
    DLOG(WARNING) << "sensor timestamp is not increasing";
    return;
  }
  if (!imu_data_queue_.empty() && imu_msg.timestamp - imu_data_queue_.back().timestamp > kMaxSensorTimestampGap) {
    DLOG(FATAL) << "IMU timestamp gap is too large: " << imu_msg.timestamp - imu_data_queue_.back().timestamp;
  }
  imu_data_queue_.push_back(imu_msg);
}

void MsgPackSynchronizer::AddEncoderData(const EncoderMsg& encoder_msg) {
  DCHECK(has_encoder_);
  if (!encoder_data_queue_.empty() && encoder_msg.timestamp <= encoder_data_queue_.back().timestamp) {
    DLOG(WARNING) << "sensor timestamp is not increasing";
    return;
  }
  if (!encoder_data_queue_.empty() && encoder_msg.timestamp - encoder_data_queue_.back().timestamp > kMaxSensorTimestampGap) {
    DLOG(FATAL) << "Encoder timestamp gap is too large: " << encoder_msg.timestamp - encoder_data_queue_.back().timestamp;
  }
  encoder_data_queue_.push_back(encoder_msg);
}

/*
  The timestamps of msg_pack points are p_0, p_1, ..., p_{N-1}.
  The timestamps of IMU are i_0, i_1, ..., i_{M-1}.
  The timestamps of the encoder are e_0, e_1, ..., e_{K-1}.

  Requirements:
  If any queue is empty, return failure.
  If the common time interval is less than 0.1s + 0.05s, return failure.
  During the first synchronization, the start times of the sensors may be asynchronous, so the sensor data should be truncated to the latest common
  time.
  group_start_time <= p_i.timestamp < group_end_time, where group_end_time == group_start_time + 0.1, and the timestamps of two consecutive
  groups are contiguous.
  group_start_time <= p_0             p_{N-1} < group_end_time.
  i_0 <= group_start_time < i_1       i_{M-2} <= group_end_time < i_{M-1}.
  e_0 <= group_start_time < e_1       e_{K-2} <= group_end_time < e_{K-1}.
*/
bool MsgPackSynchronizer::SyncMsgPack(MsgPack& msg_pack) {
  msg_pack.lidar_points.reset(new PointCloud());
  msg_pack.imu_msgs.clear();
  msg_pack.encoder_msgs.clear();

  if (lidar_data_queue_.empty() || imu_data_queue_.empty() || (has_encoder_ && encoder_data_queue_.empty())) {
    return false;
  }

  double common_begin_time = this->GetCommonBeginTime();
  double common_end_time   = this->GetCommonEndTime();
  if (common_end_time - common_begin_time < kMsgPackDuration + kMsgPackDurationPadding) {
    return false;
  }

  if (is_first_msg_pack_) {
    // ensure first lidar point is ahead of other sensors
    while (!lidar_data_queue_.empty()) {
      if (lidar_data_queue_.front().timestamp > common_begin_time) {
        break;
      }
      lidar_data_queue_.pop_front();
    }
  }

  msg_pack.group_start_time = is_first_msg_pack_ ? lidar_data_queue_.front().timestamp : last_group_end_time_;
  msg_pack.group_end_time   = msg_pack.group_start_time + kMsgPackDuration;
  msg_pack.id               = msg_pack_id_++;
  last_group_end_time_      = msg_pack.group_end_time;
  is_first_msg_pack_        = false;

  CollectLidarData(msg_pack.lidar_points, lidar_data_queue_, msg_pack.group_start_time, msg_pack.group_end_time);
  CollectImuOrEncoderData(msg_pack.imu_msgs, imu_data_queue_, msg_pack.group_start_time, msg_pack.group_end_time);
  if (has_encoder_) {
    CollectImuOrEncoderData(msg_pack.encoder_msgs, encoder_data_queue_, msg_pack.group_start_time, msg_pack.group_end_time);
  }

  ValidateMsgPack(msg_pack);

  PrintMsgPack(msg_pack);

  return true;
}

void MsgPackSynchronizer::ValidateMsgPack(const MsgPack& msg_pack) const {
  CHECK_GE(msg_pack.lidar_points->points.size(), 2);
  CHECK_GE(msg_pack.lidar_points->points.front().timestamp, msg_pack.group_start_time);
  CHECK_LT(msg_pack.lidar_points->points.back().timestamp, msg_pack.group_end_time);

  CHECK_GE(msg_pack.imu_msgs.size(), 2);
  CHECK_LE(msg_pack.imu_msgs[0].timestamp, msg_pack.group_start_time);
  CHECK_GT(msg_pack.imu_msgs[1].timestamp, msg_pack.group_start_time);
  CHECK_LE(msg_pack.imu_msgs[msg_pack.imu_msgs.size() - 2].timestamp, msg_pack.group_end_time);
  CHECK_GT(msg_pack.imu_msgs[msg_pack.imu_msgs.size() - 1].timestamp, msg_pack.group_end_time);

  if (has_encoder_) {
    CHECK_GE(msg_pack.encoder_msgs.size(), 2);
    CHECK_LE(msg_pack.encoder_msgs[0].timestamp, msg_pack.group_start_time);
    CHECK_GT(msg_pack.encoder_msgs[1].timestamp, msg_pack.group_start_time);
    CHECK_LE(msg_pack.encoder_msgs[msg_pack.encoder_msgs.size() - 2].timestamp, msg_pack.group_end_time);
    CHECK_GT(msg_pack.encoder_msgs[msg_pack.encoder_msgs.size() - 1].timestamp, msg_pack.group_end_time);
  }
}

void MsgPackSynchronizer::PrintMsgPack(const MsgPack& msg_pack) const {
  if (has_encoder_) {
    DLOG(INFO) << std::fixed << std::setprecision(6) << "msg_pack(" << msg_pack.id << ")[" << msg_pack.group_start_time << ","
              << msg_pack.group_end_time << "] lidar_" << msg_pack.lidar_points->points.size() << "["
              << msg_pack.lidar_points->points.front().timestamp << "," << msg_pack.lidar_points->points.back().timestamp << "] imu_"
              << msg_pack.imu_msgs.size() << "[" << msg_pack.imu_msgs.front().timestamp << "," << msg_pack.imu_msgs.back().timestamp << "] encoder_"
              << msg_pack.encoder_msgs.size() << "[" << msg_pack.encoder_msgs.front().timestamp << "," << msg_pack.encoder_msgs.back().timestamp
              << "]";
  } else {
    DLOG(INFO) << std::fixed << std::setprecision(6) << "msg_pack(" << msg_pack.id << ")[" << msg_pack.group_start_time << ","
              << msg_pack.group_end_time << "] lidar_" << msg_pack.lidar_points->points.size() << "["
              << msg_pack.lidar_points->points.front().timestamp << "," << msg_pack.lidar_points->points.back().timestamp << "] imu_"
              << msg_pack.imu_msgs.size() << "[" << msg_pack.imu_msgs.front().timestamp << "," << msg_pack.imu_msgs.back().timestamp << "]";
  }
}
