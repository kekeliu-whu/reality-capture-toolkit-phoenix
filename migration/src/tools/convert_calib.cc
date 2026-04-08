#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <string>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"

DEFINE_string(calib_filename, "D:\\ProjectX\\project-3d\\data\\manifold-tech-calib\\ikalibr_param.yaml", "Calibration filename");
DEFINE_string(output_dir, "D:\\output", "Output dir to save converted calibration data");

namespace {

void LoadLidarCalibration(const YAML::Node& extri, const YAML::Node& temporal, proto::SensorCalib* calib) {
  try {
    for (const auto& item : extri["SO3_LkToBr"]) {
      if (item["key"].as<std::string>() == "/livox/lidar") {
        const auto& quat   = item["value"];
        auto lidar_to_body = calib->mutable_lidar_to_encoder();
        lidar_to_body->set_rx(quat["qx"].as<double>());
        lidar_to_body->set_ry(quat["qy"].as<double>());
        lidar_to_body->set_rz(quat["qz"].as<double>());
        lidar_to_body->set_rw(quat["qw"].as<double>());
        spdlog::info("Loaded Lidar rotation: qx={}, qy={}, qz={}, qw={}", quat["qx"].as<double>(), quat["qy"].as<double>(), quat["qz"].as<double>(),
                     quat["qw"].as<double>());
        break;
      }
    }
  } catch (const std::exception& e) {
    spdlog::warn("Failed to read SO3_LkToBr: {}", e.what());
  }

  try {
    for (const auto& item : extri["POS_LkInBr"]) {
      if (item["key"].as<std::string>() == "/livox/lidar") {
        const auto& pos    = item["value"];
        auto lidar_to_body = calib->mutable_lidar_to_encoder();
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

  try {
    for (const auto& item : temporal["TO_LkToBr"]) {
      if (item["key"].as<std::string>() == "/livox/lidar") {
        double time_offset = item["value"].as<double>();
        auto lidar_to_body = calib->mutable_lidar_to_encoder();
        lidar_to_body->set_time_offset(time_offset);
        spdlog::info("Loaded Lidar time offset: {}", time_offset);
        break;
      }
    }
  } catch (const std::exception& e) {
    spdlog::warn("Failed to read TO_LkToBr: {}", e.what());
  }
}

void LoadCameraCalibration(const YAML::Node& calib_param, proto::SensorCalib* calib) {
  try {
    const auto& extri       = calib_param["EXTRI"];
    const auto& temporal    = calib_param["TEMPORAL"];
    const auto& so3_list    = extri["SO3_CmToBr"];
    const auto& pos_list    = extri["POS_CmInBr"];
    const auto& to_cm_list  = temporal["TO_CmToBr"];
    const auto& camera_list = calib_param["INTRI"]["Camera"];

    for (const auto& so3_item : so3_list) {
      try {
        std::string cam_key = so3_item["key"].as<std::string>();
        spdlog::info("Processing camera: {}", cam_key);

        auto cam_param = calib->add_camera_param();
        cam_param->set_name(cam_key);

        try {
          for (const auto& pos_item : pos_list) {
            if (pos_item["key"].as<std::string>() == cam_key) {
              const auto& quat = so3_item["value"];
              const auto& pos  = pos_item["value"];
              auto extrinsic   = cam_param->mutable_extrinsic();
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

        try {
          for (const auto& cam_item : camera_list) {
            if (cam_item["key"].as<std::string>() == cam_key) {
              const auto& cam_data = cam_item["value"]["ptr_wrapper"]["data"];
              auto focal           = cam_data["focal_length"];
              auto principal       = cam_data["principal_point"];
              auto disto           = cam_data["disto_param"];
              cam_param->set_fx(focal[0].as<double>());
              cam_param->set_fy(focal[1].as<double>());
              cam_param->set_cx(principal[0].as<double>());
              cam_param->set_cy(principal[1].as<double>());
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
}

bool ConvertCalibrationFile(const std::string& calib_filename, const std::string& output_dir) {
  proto::SensorCalib calib;

  try {
    YAML::Node yaml_root    = YAML::LoadFile(calib_filename);
    const auto& calib_param = yaml_root["CalibParam"];
    const auto& extri       = calib_param["EXTRI"];
    const auto& temporal    = calib_param["TEMPORAL"];

    LoadLidarCalibration(extri, temporal, &calib);
    LoadCameraCalibration(calib_param, &calib);
  } catch (const std::exception& e) {
    spdlog::error("Failed to load calibration file: {}", e.what());
    return false;
  }

  const std::string output_file = output_dir + "/calibration.dat";
  if (!WriteSensorCalibFile(output_file, calib)) {
    spdlog::error("Failed to save calibration file to {}", output_file);
    return false;
  }

  spdlog::info("Calibration file saved to: {}", output_file);
  return true;
}

}  // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!std::filesystem::exists(FLAGS_output_dir)) {
    spdlog::info("Output directory does not exist. Creating: {}", FLAGS_output_dir);
    std::filesystem::create_directories(FLAGS_output_dir);
  }

  if (!ConvertCalibrationFile(FLAGS_calib_filename, FLAGS_output_dir)) {
    return 1;
  }

  spdlog::info("done.");
  return 0;
}