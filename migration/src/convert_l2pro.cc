#include <custom_msgs/LixelAnyData.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/PointCloud2.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(hbc_filename, "/root/2025-05-14-183253.hbc", "Point cloud filename");
DEFINE_string(output_dir, "/root/output_dir", "Output dir to save converted data");

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

int GetFieldOffset(const sensor_msgs::PointCloud2ConstPtr& msg, const std::string& field_name) {
  for (const auto& field : msg->fields) {
    if (field.name == field_name) return field.offset;
  }
  return -1;
}

void PointCloudCallback(const sensor_msgs::PointCloud2ConstPtr& msg, std::shared_ptr<proto::LidarMsg>& lidar_msg) {
  const auto& fields = msg->fields;

  int x_offset         = GetFieldOffset(msg, "x");
  int y_offset         = GetFieldOffset(msg, "y");
  int z_offset         = GetFieldOffset(msg, "z");
  int intensity_offset = GetFieldOffset(msg, "intensity");
  int timestamp_offset = GetFieldOffset(msg, "timestamp");

  if (x_offset < 0 || y_offset < 0 || z_offset < 0 || intensity_offset < 0 || timestamp_offset < 0) {
    DLOG(INFO) << "Missing one or more fields: x/y/z/intensity/timestamp";
    return;
  }

  size_t num_points       = msg->width * msg->height;
  const uint8_t* data_ptr = msg->data.data();

  for (size_t i = 0; i < num_points; ++i) {
    float x, y, z, intensity;
    double timestamp;

    std::memcpy(&x, data_ptr + i * msg->point_step + x_offset, sizeof(float));
    std::memcpy(&y, data_ptr + i * msg->point_step + y_offset, sizeof(float));
    std::memcpy(&z, data_ptr + i * msg->point_step + z_offset, sizeof(float));
    std::memcpy(&intensity, data_ptr + i * msg->point_step + intensity_offset, sizeof(float));
    std::memcpy(&timestamp, data_ptr + i * msg->point_step + timestamp_offset, sizeof(double));

    auto point = lidar_msg->add_points();
    point->set_x(x);
    point->set_y(y);
    point->set_z(z);
    point->set_intensity(intensity);
    point->set_timestamp(timestamp);
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = 1;

  rosbag::Bag bag;
  bag.open(FLAGS_hbc_filename, rosbag::bagmode::Read);

  proto::SensorCalib sc;
  proto::ImuMsgList imu_msg_list;
  proto::EncoderMsgList encoder_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  sc.set_has_encoder(true);

  for (const auto& m : rosbag::View(bag)) {
    if (m.getTopic().find("_yaml") != std::string::npos) {
      auto msg = m.instantiate<custom_msg_pkg::LixelAnyData>();
      DCHECK(msg);

      if (m.getTopic() == "/config/extrinsic_imu_motor_yaml") {
        YAML::Node config = YAML::Load(msg->data);

        Eigen::Matrix<double, 4, 4> transform = ReadMatrix<4, 4>(config["transform"].as<std::vector<double>>());
        Eigen::Quaterniond rot(transform.block<3, 3>(0, 0));
        Eigen::Vector3d pos(transform.block<3, 1>(0, 3));
        sc.mutable_encoder_to_imu()->set_rw(rot.w());
        sc.mutable_encoder_to_imu()->set_rx(rot.x());
        sc.mutable_encoder_to_imu()->set_ry(rot.y());
        sc.mutable_encoder_to_imu()->set_rz(rot.z());
        sc.mutable_encoder_to_imu()->set_tx(pos.x());
        sc.mutable_encoder_to_imu()->set_ty(pos.y());
        sc.mutable_encoder_to_imu()->set_tz(pos.z());
      } else if (m.getTopic() == "/config/extrinsic_motor_lidar_yaml") {
        YAML::Node config = YAML::Load(msg->data);

        Eigen::Matrix<double, 4, 4> transform = ReadMatrix<4, 4>(config["transform"].as<std::vector<double>>());
        Eigen::Quaterniond rot(transform.block<3, 3>(0, 0));
        Eigen::Vector3d pos(transform.block<3, 1>(0, 3));
        sc.mutable_lidar_to_encoder()->set_rw(rot.w());
        sc.mutable_lidar_to_encoder()->set_rx(rot.x());
        sc.mutable_lidar_to_encoder()->set_ry(rot.y());
        sc.mutable_lidar_to_encoder()->set_rz(rot.z());
        sc.mutable_lidar_to_encoder()->set_tx(pos.x());
        sc.mutable_lidar_to_encoder()->set_ty(pos.y());
        sc.mutable_lidar_to_encoder()->set_tz(pos.z());
      } else if (m.getTopic() == "/config/imu_yaml") {
        YAML::Node config   = YAML::Load(msg->data);
        auto imu_instrinsic = new proto::ImuInstrinsicTpmIcra2014;
        ReadMatrixToProto<3, 3>(config["Ta"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_ta()->mutable_data());
        ReadMatrixToProto<3, 3>(config["Ka"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_ka()->mutable_data());
        ReadMatrixToProto<3, 1>(config["Ba"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_ba()->mutable_data());
        ReadMatrixToProto<3, 3>(config["Tg"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_tg()->mutable_data());
        ReadMatrixToProto<3, 3>(config["Kg"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_kg()->mutable_data());
        ReadMatrixToProto<3, 1>(config["Bg"], sc.mutable_imu_instrinsic()->mutable_tpm()->mutable_bg()->mutable_data());
      } else if (m.getTopic() == "/config/lidar_yaml") {
        YAML::Node config     = YAML::Load(msg->data);
        auto lidar_instrinsic = new proto::LidarInstrinsicSimple;
        sc.mutable_lidar_instrinsic()->mutable_simple()->set_e(config["e"].as<double>());
        ReadMatrixToProto<2, 1>(config["s"], sc.mutable_lidar_instrinsic()->mutable_simple()->mutable_s());
      }

      // DLOG(INFO) << m.getTopic();
      // DLOG(INFO) << msg->data;
    } else if (m.getTopic() == "/imu") {
      auto msg = m.instantiate<sensor_msgs::Imu>();

      auto new_msg = imu_msg_list.add_imu_msgs();
      new_msg->set_timestamp(msg->header.stamp.toSec());
      new_msg->set_gx(msg->angular_velocity.x);
      new_msg->set_gy(msg->angular_velocity.y);
      new_msg->set_gz(msg->angular_velocity.z);
      new_msg->set_ax(msg->linear_acceleration.x);
      new_msg->set_ay(msg->linear_acceleration.y);
      new_msg->set_az(msg->linear_acceleration.z);
    } else if (m.getTopic() == "/hesai/pandar") {
      auto msg = m.instantiate<sensor_msgs::PointCloud2>();

      std::shared_ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg};
      PointCloudCallback(msg, lidar_msg);
      lidar_writer.Write(lidar_msg);
    } else if (m.getTopic() == "/lidar_to_platform") {
      auto msg = m.instantiate<nav_msgs::Odometry>();

      auto new_msg = encoder_msg_list.add_encoder_msgs();
      new_msg->set_timestamp(msg->header.stamp.toSec());
      new_msg->set_rw(msg->pose.pose.orientation.w);
      new_msg->set_rx(msg->pose.pose.orientation.x);
      new_msg->set_ry(msg->pose.pose.orientation.y);
      new_msg->set_rz(msg->pose.pose.orientation.z);
    }
  }
  WriteSensorCalibFile(FLAGS_output_dir + "/calibration.dat", sc);
  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);
  WriteEncoderFile(FLAGS_output_dir + "/encoder.dat", encoder_msg_list);

  DLOG(INFO) << "done.";
}