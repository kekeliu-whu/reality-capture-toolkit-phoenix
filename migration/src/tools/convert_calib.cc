#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>

#include <filesystem>
#include <stdexcept>
#include <string>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"

DEFINE_string(calib_filename, R"(D:\calibfile\ikalibr_param_22260500001.yaml)", "Calibration filename");
DEFINE_string(output_dir, R"(D:\calibfile\output-jiuzhou2)", "Output dir to save converted calibration data");
DEFINE_string(camera_model, "OPENCV_FISHEYE", "Camera model: OPENCV or OPENCV_FISHEYE");

namespace {

namespace calibration = reality_capture::calibration;

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

void SetTransform(const YAML::Node& quat, const YAML::Node& pos, double time_offset_seconds,
                  calibration::SpatiotemporalTransform* transform) {
  auto* rotation = transform->mutable_rotation();
  rotation->set_x(RequireNode(quat, "qx").as<double>());
  rotation->set_y(RequireNode(quat, "qy").as<double>());
  rotation->set_z(RequireNode(quat, "qz").as<double>());
  rotation->set_w(RequireNode(quat, "qw").as<double>());

  auto* translation = transform->mutable_translation();
  translation->set_x(RequireNode(pos, "r0c0").as<double>());
  translation->set_y(RequireNode(pos, "r1c0").as<double>());
  translation->set_z(RequireNode(pos, "r2c0").as<double>());
  transform->set_time_offset_seconds(time_offset_seconds);
}

void SetCameraModel(const YAML::Node& focal, const YAML::Node& principal, const YAML::Node& distortion,
                    calibration::CameraCalibration* camera) {
  if (FLAGS_camera_model == "OPENCV") {
    auto* model = camera->mutable_opencv();
    model->set_focal_length_x(focal[0].as<double>());
    model->set_focal_length_y(focal[1].as<double>());
    model->set_principal_point_x(principal[0].as<double>());
    model->set_principal_point_y(principal[1].as<double>());
    model->set_k1(distortion[0].as<double>());
    model->set_k2(distortion[1].as<double>());
    model->set_p1(distortion[2].as<double>());
    model->set_p2(distortion[3].as<double>());
    return;
  }
  if (FLAGS_camera_model == "OPENCV_FISHEYE") {
    auto* model = camera->mutable_opencv_fisheye();
    model->set_focal_length_x(focal[0].as<double>());
    model->set_focal_length_y(focal[1].as<double>());
    model->set_principal_point_x(principal[0].as<double>());
    model->set_principal_point_y(principal[1].as<double>());
    model->set_k1(distortion[0].as<double>());
    model->set_k2(distortion[1].as<double>());
    model->set_k3(distortion[2].as<double>());
    model->set_k4(distortion[3].as<double>());
    return;
  }
  throw std::runtime_error("Unsupported camera model: " + FLAGS_camera_model);
}

void LoadLidarCalibration(const YAML::Node& extri, const YAML::Node& temporal, calibration::SensorCalibration* calib) {
  const auto quat          = FindItemByKey(RequireNode(extri, "SO3_LkToBr"), "/livox/lidar", "EXTRI.SO3_LkToBr");
  const auto pos           = FindItemByKey(RequireNode(extri, "POS_LkInBr"), "/livox/lidar", "EXTRI.POS_LkInBr");
  const auto time_offset   = FindItemByKey(RequireNode(temporal, "TO_LkToBr"), "/livox/lidar", "TEMPORAL.TO_LkToBr");
  auto* imu_from_lidar = calib->mutable_direct()->mutable_imu_from_lidar();
  SetTransform(quat, pos, time_offset.as<double>(), imu_from_lidar);

  const auto& rotation = imu_from_lidar->rotation();
  const auto& translation = imu_from_lidar->translation();
  spdlog::info("Loaded Lidar rotation: qx={}, qy={}, qz={}, qw={}", rotation.x(), rotation.y(), rotation.z(), rotation.w());
  spdlog::info("Loaded Lidar position: x={}, y={}, z={}", translation.x(), translation.y(), translation.z());
  spdlog::info("Loaded Lidar time offset: {}", imu_from_lidar->time_offset_seconds());
}

void LoadCameraCalibration(const YAML::Node& calib_param, calibration::SensorCalibration* calib) {
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

    auto* camera = calib->add_cameras();
    camera->set_name(cam_key);

    auto* imu_from_camera = camera->mutable_imu_from_camera();
    SetTransform(quat, pos, time_offset.as<double>(), imu_from_camera);

    const auto focal     = RequireNode(cam_data, "focal_length");
    const auto principal = RequireNode(cam_data, "principal_point");
    const auto disto     = RequireNode(cam_data, "disto_param");
    if (!focal.IsSequence() || focal.size() < 2 || !principal.IsSequence() || principal.size() < 2 || !disto.IsSequence() || disto.size() < 4) {
      throw std::runtime_error("Camera intrinsic array size is invalid for " + cam_key);
    }

    SetCameraModel(focal, principal, disto, camera);

    const auto& rotation = imu_from_camera->rotation();
    const auto& translation = imu_from_camera->translation();
    spdlog::info("{} extrinsic: rot({}, {}, {}, {}), trans({}, {}, {})", cam_key, rotation.x(), rotation.y(), rotation.z(), rotation.w(),
                 translation.x(), translation.y(), translation.z());
    spdlog::info("{} time offset: {}", cam_key, imu_from_camera->time_offset_seconds());
    spdlog::info("{} camera model: {}", cam_key, FLAGS_camera_model);
  }
}

bool ConvertCalibrationFile(const std::string& calib_filename, const std::string& output_dir) {
  calibration::SensorCalibration calib;

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
  if (!WriteSensorCalibrationFile(output_file, calib)) {
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
