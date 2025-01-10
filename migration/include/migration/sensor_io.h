#pragma once

#include <fstream>
#include <functional>

#include "proto/sensors.pb.h"

template <typename T> using Ptr = std::shared_ptr<T>;
template <typename T> using ConstPtr = std::shared_ptr<const T>;

bool ReadLidarFile(const std::string &filename,
                   std::function<void(const ConstPtr<LidarMsg> &)> callback);
bool WriteLidarFile(const std::string &filename,
                    const std::vector<ConstPtr<LidarMsg>> &scans);

bool ReadUndistortedLidarFile(
    const std::string &filename,
    std::function<void(const ConstPtr<UndistoredLidarMsg> &)> callback);
class LidarFileWriter {
public:
  LidarFileWriter();

  bool Open(const std::string &filename);

  bool Write(const ConstPtr<UndistoredLidarMsg> &msg);

  void Close();

  ~LidarFileWriter();

private:
  std::string filename_;
  std::ofstream outfile_;
};

bool ReadImuFile(const std::string &filename, ImuMsgList &imu);
bool WriteImuFile(const std::string &filename, const ImuMsgList &imu);

bool ReadMotorFile(const std::string &filename, MotorMsgList &motor);
bool WriteMotorFile(const std::string &filename, const MotorMsgList &motor);

bool ReadPoseFile(const std::string &filename, PoseMsgList &pose);
bool WritePoseFile(const std::string &filename, const PoseMsgList &pose);

bool ReadPgoConfigFile(const std::string &filename, PgoConfig &config);
