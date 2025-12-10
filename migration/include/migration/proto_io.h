#pragma once

#include <spdlog/spdlog.h>

#include <glog/logging.h>
#include <fstream>
#include <functional>

#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

template <typename T>
using Ptr = std::shared_ptr<T>;
template <typename T>
using ConstPtr = std::shared_ptr<const T>;

bool ReadLidarFile(const std::string &filename, std::function<void(const ConstPtr<proto::LidarMsg> &)> callback);
bool WriteLidarFile(const std::string &filename, const std::vector<ConstPtr<proto::LidarMsg>> &scans);

bool ReadUndistortedLidarFile(const std::string &filename, std::function<void(const ConstPtr<proto::UndistoredLidarMsg> &)> callback);

bool ReadImuFile(const std::string &filename, proto::ImuMsgList &imu);
bool WriteImuFile(const std::string &filename, const proto::ImuMsgList &imu);

bool ReadEncoderFile(const std::string &filename, proto::EncoderMsgList &motor);
bool WriteEncoderFile(const std::string &filename, const proto::EncoderMsgList &motor);

bool ReadPoseFile(const std::string &filename, proto::PoseMsgList &pose);
bool WritePoseFile(const std::string &filename, const proto::PoseMsgList &pose);

bool ReadPgoConfigFile(const std::string &filename, proto::PgoConfig &config);

bool ReadSensorCalibFile(const std::string &filename, proto::SensorCalib &calib);
bool WriteSensorCalibFile(const std::string &filename, const proto::SensorCalib &calib);

bool WriteDelimitedTo(const google::protobuf::MessageLite &message, std::ofstream &rawOutput);

template <typename T>
class SequentialLidarFileWriter {
 public:
  SequentialLidarFileWriter() {}

  bool Open(const std::string &filename) {
    filename_ = filename;
    outfile_.open(filename, std::ios::out | std::ios::binary);
    LOG_IF(INFO, !outfile_.is_open()) << "Create lidar file: " << filename;
    return outfile_.is_open();
  }

  bool Write(const ConstPtr<T> &msg) {
    if (!WriteDelimitedTo(*msg, outfile_)) {
      spdlog::debug("Failed to serialize proto::UndistoredLidarMsg to file: {}", filename_);
      return false;
    }
    return true;
  }

  void Close() { outfile_.close(); }

  ~SequentialLidarFileWriter() { Close(); }

 private:
  std::string filename_;
  std::ofstream outfile_;
};
