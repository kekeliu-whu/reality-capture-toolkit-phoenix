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

bool WriteLidarFile(const std::string &filename, const std::vector<ConstPtr<proto::LidarMsg>> &scans);

bool ReadUndistortedLidarFile(const std::string &filename, std::function<void(const ConstPtr<proto::UndistoredLidarMsg> &)> callback);

bool ReadImuFile(const std::string &filename, proto::ImuMsgList &imu);
bool WriteImuFile(const std::string &filename, const proto::ImuMsgList &imu);

bool ReadEncoderFile(const std::string &filename, proto::EncoderMsgList &motor);
bool WriteEncoderFile(const std::string &filename, const proto::EncoderMsgList &motor);

bool ReadGnssFile(const std::string &filename, proto::GpsMsgList &gnss);
bool WriteGnssFile(const std::string &filename, const proto::GpsMsgList &gnss);

bool ReadPoseFile(const std::string &filename, proto::PoseMsgList &pose);
bool WritePoseFile(const std::string &filename, const proto::PoseMsgList &pose);

bool ReadPgoConfigFile(const std::string &filename, proto::PgoConfig &config);

bool ReadSensorCalibFile(const std::string &filename, proto::SensorCalib &calib);
bool WriteSensorCalibFile(const std::string &filename, const proto::SensorCalib &calib);

bool WriteDelimitedTo(const google::protobuf::MessageLite &message, std::ofstream &rawOutput);
bool ReadDelimitedFrom(std::ifstream &rawInput, google::protobuf::MessageLite *message);

template <typename T>
class SequentialLidarFileReader {
 public:
  SequentialLidarFileReader() : file_ended_(false), file_size_(0) {}

  bool Open(const std::string &filename) {
    filename_ = filename;
    infile_.open(filename, std::ios::in | std::ios::binary);
    if (!infile_.is_open()) {
      spdlog::warn("Failed to open lidar file: {}", filename);
      return false;
    }

    // Get file size
    infile_.seekg(0, std::ios::end);
    file_size_ = infile_.tellg();
    infile_.seekg(0, std::ios::beg);

    spdlog::info("Opened lidar file: {} (size: {} bytes)", filename, file_size_);
    return true;
  }

  bool ReadNext(Ptr<T> &msg) {
    if (file_ended_) {
      spdlog::warn("End of lidar file reached: {}", filename_);
      return false;
    }

    if (!infile_.is_open()) {
      spdlog::warn("Lidar file is not open: {}", filename_);
      return false;
    }

    msg = std::make_shared<T>();
    if (!ReadDelimitedFrom(infile_, msg.get())) {
      file_ended_ = true;
      return false;
    }
    return true;
  }

  bool IsFileEnded() const { return file_ended_; }

  double getProgress() {
    if (file_size_ <= 0) {
      return 0.0;
    }
    std::streampos current_pos = infile_.tellg();
    return (current_pos / (double)file_size_) * 100.0;
  }

  void Close() {
    if (infile_.is_open()) {
      infile_.close();
    }
  }

  ~SequentialLidarFileReader() { Close(); }

 private:
  std::string filename_;
  std::ifstream infile_;
  bool file_ended_;
  std::streamsize file_size_;
};

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
      spdlog::info("Failed to serialize proto::UndistoredLidarMsg to file: {}", filename_);
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
