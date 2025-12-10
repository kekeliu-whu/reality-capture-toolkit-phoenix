#pragma once

#include <spdlog/spdlog.h>
#include <algorithm>
#include <deque>

#include "common/types.h"

class MsgPackSynchronizer {
 public:
  MsgPackSynchronizer(bool has_encoder) : has_encoder_(has_encoder) {}

  void AddLidarData(const LidarMsg::ConstPtr& lidar_msg);
  void AddImuData(const ImuMsg& imu_msg);
  void AddEncoderData(const EncoderMsg& encoder_msg);

  bool SyncMsgPack(MsgPack& msg_pack);

 private:
  double GetCommonBeginTime() const {
    if (has_encoder_) {
      spdlog::debug("Checking non-empty queues for lidar, imu, and encoder data");
      return std::max({lidar_data_queue_.front().timestamp, imu_data_queue_.front().timestamp, encoder_data_queue_.front().timestamp});
    } else {
      spdlog::debug("Checking non-empty queues for lidar and imu data");
      return std::max(lidar_data_queue_.front().timestamp, imu_data_queue_.front().timestamp);
    }
  }

  double GetCommonEndTime() const {
    if (has_encoder_) {
      spdlog::debug("Checking non-empty queues for lidar, imu, and encoder data");
      return std::min({lidar_data_queue_.back().timestamp, imu_data_queue_.back().timestamp, encoder_data_queue_.back().timestamp});
    } else {
      spdlog::debug("Checking non-empty queues for lidar and imu data");
      return std::min(lidar_data_queue_.back().timestamp, imu_data_queue_.back().timestamp);
    }
  }

 private:
  void ValidateMsgPack(const MsgPack& msg_pack) const;

  void PrintMsgPack(const MsgPack& msg_pack) const;

 private:
  bool has_encoder_;

  std::deque<PointXYZIRT> lidar_data_queue_;
  std::deque<ImuMsg> imu_data_queue_;
  std::deque<EncoderMsg> encoder_data_queue_;

  bool is_first_msg_pack_ = true;
  int msg_pack_id_ = 0;
  double last_group_end_time_;

  static constexpr double kMaxSensorTimestampGap  = 0.3;
  static constexpr double kMsgPackDuration        = 0.1;
  static constexpr double kMsgPackDurationPadding = 0.05;
};
