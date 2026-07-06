#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"

DEFINE_string(calib_filename, R"(D:\calibfile\ikalibr_param_22260500001.yaml)", "Calibration filename");
DEFINE_string(output_dir, R"(D:\calibfile\output-jiuzhou2)", "Output dir to save converted calibration data");

namespace {

YAML::Node RequireNode(const YAML::Node& parent, const char* key) {
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::runtime_error(std::string("Missing required node: ") + key);
  }
  return node;
}

const YAML::Node FindItemByKey(const YAML::Node& sequence, const std::string& key, const char* sequence_name) {
  if (!sequence || !sequence.IsSequence()) {
    throw std::runtime_error(std::string("Expected sequence node: ") + sequence_name);
  }

  for (const auto& item : sequence) {
    if (RequireNode(item, "key").as<std::string>() == key) {
      return RequireNode(item, "value");
    }
  }

  throw std::runtime_error(std::string("Missing required entry '") + key + "' in " + sequence_name);
}

void LoadLidarCalibration(const YAML::Node& extri, const YAML::Node& temporal, proto::SensorCalib* calib) {
  const auto quat          = FindItemByKey(RequireNode(extri, "SO3_LkToBr"), "/livox/lidar", "EXTRI.SO3_LkToBr");
  const auto pos           = FindItemByKey(RequireNode(extri, "POS_LkInBr"), "/livox/lidar", "EXTRI.POS_LkInBr");
  const auto time_offset   = FindItemByKey(RequireNode(temporal, "TO_LkToBr"), "/livox/lidar", "TEMPORAL.TO_LkToBr");
  auto lidar_to_body       = calib->mutable_lidar_to_encoder();
  lidar_to_body->set_rx(RequireNode(quat, "qx").as<double>());
  lidar_to_body->set_ry(RequireNode(quat, "qy").as<double>());
  lidar_to_body->set_rz(RequireNode(quat, "qz").as<double>());
  lidar_to_body->set_rw(RequireNode(quat, "qw").as<double>());
  lidar_to_body->set_tx(RequireNode(pos, "r0c0").as<double>());
  lidar_to_body->set_ty(RequireNode(pos, "r1c0").as<double>());
  lidar_to_body->set_tz(RequireNode(pos, "r2c0").as<double>());
  lidar_to_body->set_time_offset(time_offset.as<double>());

  spdlog::info("Loaded Lidar rotation: qx={}, qy={}, qz={}, qw={}", lidar_to_body->rx(), lidar_to_body->ry(), lidar_to_body->rz(),
               lidar_to_body->rw());
  spdlog::info("Loaded Lidar position: tx={}, ty={}, tz={}", lidar_to_body->tx(), lidar_to_body->ty(), lidar_to_body->tz());
  spdlog::info("Loaded Lidar time offset: {}", lidar_to_body->time_offset());
}

void LoadCameraCalibration(const YAML::Node& calib_param, proto::SensorCalib* calib) {
  const auto& extri       = RequireNode(calib_param, "EXTRI");
  const auto& temporal    = RequireNode(calib_param, "TEMPORAL");
  const auto& so3_list    = RequireNode(extri, "SO3_CmToBr");
  const auto& pos_list    = RequireNode(extri, "POS_CmInBr");
  const auto& to_cm_list  = RequireNode(temporal, "TO_CmToBr");
  const auto& intri       = RequireNode(calib_param, "INTRI");
  const auto& camera_list = RequireNode(intri, "Camera");

  if (!so3_list.IsSequence() || so3_list.size() == 0) {
    throw std::runtime_error("No camera extrinsics found in EXTRI.SO3_CmToBr");
  }

  for (const auto& so3_item : so3_list) {
    std::string cam_key    = RequireNode(so3_item, "key").as<std::string>();
    const auto quat        = RequireNode(so3_item, "value");
    const auto pos         = FindItemByKey(pos_list, cam_key, "EXTRI.POS_CmInBr");
    const auto time_offset = FindItemByKey(to_cm_list, cam_key, "TEMPORAL.TO_CmToBr");
    const auto cam_data    = RequireNode(RequireNode(FindItemByKey(camera_list, cam_key, "INTRI.Camera"), "ptr_wrapper"), "data");

    spdlog::info("Processing camera: {}", cam_key);

    auto cam_param = calib->add_camera_param();
    cam_param->set_name(cam_key);

    auto extrinsic = cam_param->mutable_extrinsic();
    extrinsic->set_rx(RequireNode(quat, "qx").as<double>());
    extrinsic->set_ry(RequireNode(quat, "qy").as<double>());
    extrinsic->set_rz(RequireNode(quat, "qz").as<double>());
    extrinsic->set_rw(RequireNode(quat, "qw").as<double>());
    extrinsic->set_tx(RequireNode(pos, "r0c0").as<double>());
    extrinsic->set_ty(RequireNode(pos, "r1c0").as<double>());
    extrinsic->set_tz(RequireNode(pos, "r2c0").as<double>());
    extrinsic->set_time_offset(time_offset.as<double>());

    const auto focal     = RequireNode(cam_data, "focal_length");
    const auto principal = RequireNode(cam_data, "principal_point");
    const auto disto     = RequireNode(cam_data, "disto_param");
    if (!focal.IsSequence() || focal.size() < 2 || !principal.IsSequence() || principal.size() < 2 || !disto.IsSequence() || disto.size() < 4) {
      throw std::runtime_error("Camera intrinsic array size is invalid for " + cam_key);
    }

    cam_param->set_fx(focal[0].as<double>());
    cam_param->set_fy(focal[1].as<double>());
    cam_param->set_cx(principal[0].as<double>());
    cam_param->set_cy(principal[1].as<double>());
    cam_param->set_k1(disto[0].as<double>());
    cam_param->set_k2(disto[1].as<double>());
    cam_param->set_k3(disto[2].as<double>());
    cam_param->set_k4(disto[3].as<double>());

    spdlog::info("{} extrinsic: rot({}, {}, {}, {}), trans({}, {}, {})", cam_key, extrinsic->rx(), extrinsic->ry(), extrinsic->rz(), extrinsic->rw(),
                 extrinsic->tx(), extrinsic->ty(), extrinsic->tz());
    spdlog::info("{} time offset: {}", cam_key, extrinsic->time_offset());
    spdlog::info("{} intrinsic: fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, k3={}, k4={}", cam_key, cam_param->fx(), cam_param->fy(), cam_param->cx(),
                 cam_param->cy(), cam_param->k1(), cam_param->k2(), cam_param->k3(), cam_param->k4());
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