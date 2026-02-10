#include <custom_msgs/LixelAnyData.h>
#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(bag_filename, "\\\\wsl.localhost\\Ubuntu-24.04\\home\\rick\\iKalibr\\2026-02-06_11-34-29\\all_2026-02-06-11-34-35.bag",
              "Point cloud filename");
DEFINE_string(output_dir, "D:\\slam",
              "Output dir to save converted data");

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> ReadMatrix(const std::vector<double>& mat) {
  CHECK_EQ(Rows * Cols, mat.size());
  return Eigen::Map<const Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>>{mat.data()};
}

template <int Rows, int Cols>
void ReadMatrixToProto(const YAML::Node& node, google::protobuf::RepeatedField<double>* mat_proto) {
  auto mat_vec = node.as<std::vector<double>>();
  CHECK_EQ(Rows * Cols, mat_vec.size());
  for (auto& e : mat_vec) {
    mat_proto->Add(e);
  }
}

void PointCloudCallback(const livox_ros_driver::CustomMsgConstPtr& msg, std::shared_ptr<proto::LidarMsg>& lidar_msg) {
  for (size_t i = 0; i < msg->points.size(); ++i) {
    auto& pt = msg->points[i];

    if ((pt.tag & 0x30) == 0x10 || (pt.tag & 0x30) == 0x00) {
      auto point = lidar_msg->add_points();
      point->set_x(pt.x);
      point->set_y(pt.y);
      point->set_z(pt.z);
      point->set_intensity(pt.reflectivity);
      point->set_timestamp((pt.offset_time + msg->timebase) * 1e-9);
    }
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  rosbag::Bag bag;
  bag.open(FLAGS_bag_filename, rosbag::bagmode::Read);

  proto::SensorCalib sc;
  proto::ImuMsgList imu_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  for (const auto& m : rosbag::View(bag)) {
    if (m.getTopic() == "/livox/imu") {
      auto msg = m.instantiate<sensor_msgs::Imu>();

      auto new_msg = imu_msg_list.add_imu_msgs();
      new_msg->set_timestamp(msg->header.stamp.toSec());
      new_msg->set_gx(msg->angular_velocity.x);
      new_msg->set_gy(msg->angular_velocity.y);
      new_msg->set_gz(msg->angular_velocity.z);
      new_msg->set_ax(msg->linear_acceleration.x * 9.8);
      new_msg->set_ay(msg->linear_acceleration.y * 9.8);
      new_msg->set_az(msg->linear_acceleration.z * 9.8);
    } else if (m.getTopic() == "/livox/lidar") {
      auto msg = m.instantiate<livox_ros_driver::CustomMsg>();

      std::shared_ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg};
      PointCloudCallback(msg, lidar_msg);
      lidar_writer.Write(lidar_msg);
    }
  }

  {
    sc.mutable_lidar_to_encoder()->set_tx(-0.011);
    sc.mutable_lidar_to_encoder()->set_ty(-0.02329);
    sc.mutable_lidar_to_encoder()->set_tz(0.04412);
    sc.mutable_lidar_to_encoder()->set_rw(1);
    sc.mutable_lidar_to_encoder()->set_rx(0);
    sc.mutable_lidar_to_encoder()->set_ry(0);
    sc.mutable_lidar_to_encoder()->set_rz(0);

    WriteSensorCalibFile(FLAGS_output_dir + "/calibration.dat", sc);
  }
  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);

  spdlog::info("done.");
}