#include <glog/logging.h>

#include "core/preprocess.h"

void ProcessRawSensorData(const SensorCalib& calib, MsgPack& msg_pack) {
  for (auto& e : msg_pack.imu_msgs) {
    calib.imu_instrinsic->Deskew(e.acc, e.gyr);
  }

  for (auto& e : *msg_pack.lidar_points) {
    Eigen::Map<Eigen::Vector3f> vec = e.getVector3fMap();
    calib.lidar_instrinsic->Deskew(vec);
  }

  for (auto& e : *msg_pack.lidar_points) {
    Eigen::Vector3d p = e.getVector3fMap().cast<double>();

    p = calib.lidar_to_encoder * p;

    if (calib.has_encoder) {
      int idx = std::distance(msg_pack.encoder_msgs.begin(), std::upper_bound(msg_pack.encoder_msgs.begin(), msg_pack.encoder_msgs.end(), e.timestamp,
                                                                              [](double t, const EncoderMsg& msg) { return t < msg.timestamp; }));
      CHECK_GT(idx, 0);
      CHECK_LE(idx, msg_pack.encoder_msgs.size() - 1);

      double factor = (e.timestamp - msg_pack.encoder_msgs[idx - 1].timestamp) /
                      (msg_pack.encoder_msgs[idx].timestamp - msg_pack.encoder_msgs[idx - 1].timestamp);
      CHECK_GE(factor, 0);
      CHECK_LE(factor, 1);

      Quaternion encoder_rot = msg_pack.encoder_msgs[idx - 1].rot.slerp(factor, msg_pack.encoder_msgs[idx].rot);

      p = encoder_rot * p;
      p = calib.encoder_to_imu * p;
    }

    e.getVector3fMap() = p.cast<float>();
  }
}
