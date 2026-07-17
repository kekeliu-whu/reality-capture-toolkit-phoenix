#include <gflags/gflags.h>
#include <lixel_msgs/AnyData.h>
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

namespace calibration = reality_capture::calibration;

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

void SetVector3FromYaml(const YAML::Node& node, calibration::Vector3* vector) {
  const auto values = node.as<std::vector<double>>();
  CHECK_EQ(3, values.size());
  vector->set_x(values[0]);
  vector->set_y(values[1]);
  vector->set_z(values[2]);
}

void SetTransformFromMatrix(const Eigen::Matrix4d& transform, calibration::SpatiotemporalTransform* output) {
  Eigen::Quaterniond rotation(transform.block<3, 3>(0, 0));
  rotation.normalize();
  auto* quaternion = output->mutable_rotation();
  quaternion->set_x(rotation.x());
  quaternion->set_y(rotation.y());
  quaternion->set_z(rotation.z());
  quaternion->set_w(rotation.w());

  auto* translation = output->mutable_translation();
  translation->set_x(transform(0, 3));
  translation->set_y(transform(1, 3));
  translation->set_z(transform(2, 3));
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
    spdlog::info("Missing one or more fields: x/y/z/intensity/timestamp");
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

  rosbag::Bag bag;
  bag.open(FLAGS_hbc_filename, rosbag::bagmode::Read);

  calibration::SensorCalibration sc;
  proto::ImuMsgList imu_msg_list;
  proto::EncoderMsgList encoder_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  for (const auto& m : rosbag::View(bag)) {
    if (m.getTopic().find("_yaml") != std::string::npos) {
      auto msg = m.instantiate<lixel_msgs::AnyData>();
      if (!msg) {
        spdlog::error("msg failed");
        exit(1);
      }

      if (m.getTopic() == "/config/extrinsic_imu_motor_yaml") {
        YAML::Node config = YAML::Load(msg->data);

        Eigen::Matrix<double, 4, 4> transform = ReadMatrix<4, 4>(config["transform"].as<std::vector<double>>());
        SetTransformFromMatrix(transform, sc.mutable_via_encoder()->mutable_imu_from_encoder());
      } else if (m.getTopic() == "/config/extrinsic_motor_lidar_yaml") {
        YAML::Node config = YAML::Load(msg->data);

        Eigen::Matrix<double, 4, 4> transform = ReadMatrix<4, 4>(config["transform"].as<std::vector<double>>());
        SetTransformFromMatrix(transform, sc.mutable_via_encoder()->mutable_encoder_from_lidar());
      } else if (m.getTopic() == "/config/imu_yaml") {
        YAML::Node config = YAML::Load(msg->data);
        auto* model = sc.mutable_imu_intrinsics()->mutable_tpm_icra_2014();
        ReadMatrixToProto<3, 3>(config["Ta"], model->mutable_accelerometer_ta()->mutable_row_major_values());
        ReadMatrixToProto<3, 3>(config["Ka"], model->mutable_accelerometer_ka()->mutable_row_major_values());
        SetVector3FromYaml(config["Ba"], model->mutable_accelerometer_bias());
        ReadMatrixToProto<3, 3>(config["Tg"], model->mutable_gyroscope_tg()->mutable_row_major_values());
        ReadMatrixToProto<3, 3>(config["Kg"], model->mutable_gyroscope_kg()->mutable_row_major_values());
        SetVector3FromYaml(config["Bg"], model->mutable_gyroscope_bias());
      } else if (m.getTopic() == "/config/lidar_yaml") {
        YAML::Node config = YAML::Load(msg->data);
        sc.mutable_lidar_intrinsics()->mutable_simple()->set_e(config["e"].as<double>());
        ReadMatrixToProto<2, 1>(config["s"], sc.mutable_lidar_intrinsics()->mutable_simple()->mutable_s());
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
  WriteSensorCalibrationFile(FLAGS_output_dir + "/calibration.dat", sc);
  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);
  WriteEncoderFile(FLAGS_output_dir + "/encoder.dat", encoder_msg_list);

  spdlog::info("done.");
}
