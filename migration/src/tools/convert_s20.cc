#include <custom_msgs/LixelAnyData.h>
#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>

#include <filesystem>
#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(bag_filename, "\\\\wsl.localhost\\Ubuntu-24.04\\home\\rick\\iKalibr\\src\\iKalibr\\2026-02-06_11-34-29-s20\\all_2026-02-06-11-34-35.bag",
              "Point cloud filename");
DEFINE_string(calib_filename,
              "\\\\wsl.localhost\\Ubuntu-24.04\\home\\rick\\iKalibr\\src\\iKalibr\\2026-02-06_11-34-29-s20\\ikalibr_output\\ikalibr_param.yaml",
              "Calibration filename");
DEFINE_string(output_dir, "D:\\slam", "Output dir to save converted data");

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

  if (!std::filesystem::exists(FLAGS_output_dir)) {
    spdlog::info("Output directory does not exist. Creating: {}", FLAGS_output_dir);
    std::filesystem::create_directories(FLAGS_output_dir);
  }

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

      // spdlog::info("Processed IMU message at time: {:.6f} acc: {:.3f}, {:.3f}, {:.3f} gyro: {:.3f}, {:.3f}, {:.3f}", msg->header.stamp.toSec(),
      //              msg->linear_acceleration.x * 9.8, msg->linear_acceleration.y * 9.8, msg->linear_acceleration.z * 9.8, msg->angular_velocity.x,
      //              msg->angular_velocity.y, msg->angular_velocity.z);
    } else if (m.getTopic() == "/livox/lidar") {
      auto msg = m.instantiate<livox_ros_driver::CustomMsg>();

      std::shared_ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg};
      PointCloudCallback(msg, lidar_msg);
      lidar_writer.Write(lidar_msg);
    }
  }

  {
    // Read calibration from YAML file
    try {
      YAML::Node yaml_root    = YAML::LoadFile(FLAGS_calib_filename);
      const auto& calib_param = yaml_root["CalibParam"];
      const auto& extri       = calib_param["EXTRI"];
      const auto& temporal    = calib_param["TEMPORAL"];

      // Read SO3_LkToBr (Lidar rotation relative to body frame)
      try {
        for (const auto& item : extri["SO3_LkToBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            const auto& quat   = item["value"];
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_rx(quat["qx"].as<double>());
            lidar_to_body->set_ry(quat["qy"].as<double>());
            lidar_to_body->set_rz(quat["qz"].as<double>());
            lidar_to_body->set_rw(quat["qw"].as<double>());
            spdlog::info("Loaded Lidar rotation: qx={}, qy={}, qz={}, qw={}", quat["qx"].as<double>(), quat["qy"].as<double>(),
                         quat["qz"].as<double>(), quat["qw"].as<double>());
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read SO3_LkToBr: {}", e.what());
      }

      // Read POS_LkInBr (Lidar position relative to body frame)
      try {
        for (const auto& item : extri["POS_LkInBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            const auto& pos    = item["value"];
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_tx(pos["r0c0"].as<double>());
            lidar_to_body->set_ty(pos["r1c0"].as<double>());
            lidar_to_body->set_tz(pos["r2c0"].as<double>());
            spdlog::info("Loaded Lidar position: tx={}, ty={}, tz={}", pos["r0c0"].as<double>(), pos["r1c0"].as<double>(), pos["r2c0"].as<double>());
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read POS_LkInBr: {}", e.what());
      }

      // Read TO_LkToBr (Lidar time offset relative to body frame)
      try {
        for (const auto& item : temporal["TO_LkToBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            double time_offset = item["value"].as<double>();
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_time_offset(time_offset);
            spdlog::info("Loaded Lidar time offset: {}", time_offset);
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read TO_LkToBr: {}", e.what());
      }

      // Read and output camera extrinsic and intrinsic parameters
      try {
        const auto& so3_list    = extri["SO3_CmToBr"];
        const auto& pos_list    = extri["POS_CmInBr"];
        const auto& to_cm_list  = temporal["TO_CmToBr"];
        const auto& camera_list = calib_param["INTRI"]["Camera"];

        for (const auto& so3_item : so3_list) {
          try {
            std::string cam_key = so3_item["key"].as<std::string>();
            spdlog::info("Processing camera: {}", cam_key);

            auto cam_param = sc.add_camera_param();
            cam_param->set_name(cam_key);

            // Find and set matching position entry
            try {
              for (const auto& pos_item : pos_list) {
                if (pos_item["key"].as<std::string>() == cam_key) {
                  const auto& quat = so3_item["value"];
                  const auto& pos  = pos_item["value"];

                  // Set camera to body extrinsic (rotation + translation)
                  auto extrinsic = cam_param->mutable_extrinsic();
                  extrinsic->set_rx(quat["qx"].as<double>());
                  extrinsic->set_ry(quat["qy"].as<double>());
                  extrinsic->set_rz(quat["qz"].as<double>());
                  extrinsic->set_rw(quat["qw"].as<double>());
                  extrinsic->set_tx(pos["r0c0"].as<double>());
                  extrinsic->set_ty(pos["r1c0"].as<double>());
                  extrinsic->set_tz(pos["r2c0"].as<double>());

                  spdlog::info("{} extrinsic: rot({}, {}, {}, {}), trans({}, {}, {})", cam_key, quat["qx"].as<double>(), quat["qy"].as<double>(),
                               quat["qz"].as<double>(), quat["qw"].as<double>(), pos["r0c0"].as<double>(), pos["r1c0"].as<double>(),
                               pos["r2c0"].as<double>());
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read position for camera {}: {}", cam_key, e.what());
            }

            // Find and set camera time offset
            try {
              for (const auto& time_item : to_cm_list) {
                if (time_item["key"].as<std::string>() == cam_key) {
                  double time_offset = time_item["value"].as<double>();
                  cam_param->mutable_extrinsic()->set_time_offset(time_offset);
                  spdlog::info("{} time offset: {}", cam_key, time_offset);
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read time offset for camera {}: {}", cam_key, e.what());
            }

            // Find and set camera intrinsic parameters
            try {
              for (const auto& cam_item : camera_list) {
                if (cam_item["key"].as<std::string>() == cam_key) {
                  const auto& cam_data = cam_item["value"]["ptr_wrapper"]["data"];

                  // Set focal length
                  auto focal = cam_data["focal_length"];
                  cam_param->set_fx(focal[0].as<double>());
                  cam_param->set_fy(focal[1].as<double>());

                  // Set principal point
                  auto principal = cam_data["principal_point"];
                  cam_param->set_cx(principal[0].as<double>());
                  cam_param->set_cy(principal[1].as<double>());

                  // Set distortion parameters
                  auto disto = cam_data["disto_param"];
                  cam_param->set_k1(disto[0].as<double>());
                  cam_param->set_k2(disto[1].as<double>());
                  cam_param->set_k3(disto[2].as<double>());
                  cam_param->set_k4(disto[3].as<double>());

                  spdlog::info("{} intrinsic: fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, k3={}, k4={}", cam_key, cam_param->fx(), cam_param->fy(),
                               cam_param->cx(), cam_param->cy(), cam_param->k1(), cam_param->k2(), cam_param->k3(), cam_param->k4());
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read intrinsic for camera {}: {}", cam_key, e.what());
            }
          } catch (const std::exception& e) {
            spdlog::warn("Failed to process camera: {}", e.what());
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read camera extrinsic and intrinsic parameters: {}", e.what());
      }

      WriteSensorCalibFile(FLAGS_output_dir + "/calibration.dat", sc);
      spdlog::info("Calibration file saved to: {}/calibration.dat", FLAGS_output_dir);
    } catch (const std::exception& e) {
      spdlog::error("Failed to load calibration file: {}", e.what());
    }
  }

  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);

  spdlog::info("done.");
}