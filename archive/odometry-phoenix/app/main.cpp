#include <gflags/gflags.h>
#include <glog/logging.h>
#include <spdlog/spdlog.h>
#include <yaml-cpp/yaml.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <Windows.h>
#endif

#include <pdal/Dimension.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>

#include "lio.h"
#include "method/ulog.h"
#include "migration/inc_las_writer.h"
#include "migration/proto_io.h"

DEFINE_string(input_dir,
              "",
              "Directory containing calibration.dat, imu.dat and lidar.dat");
DEFINE_string(output_dir,
              "",
              "Directory for realtime-mapping results");
DEFINE_string(config_filename,
              "L2PRO.yaml",
              "Realtime mapping YAML config. Relative paths are resolved from "
              "the executable directory.");
DEFINE_string(external_trajectory_bin,
              "",
              "Optional KTRJ trajectory used to replay and undistort lidar.dat");
DEFINE_int32(max_lidar_frames,
             0,
             "Stop after feeding this many input LiDAR frames; 0 processes all data");
DEFINE_int32(skip_lidar_frames,
             0,
             "Skip this many source LiDAR frames before dynamic initialization");
DEFINE_bool(force_dynamic_initialization,
            false,
            "Exercise the moving-start bootstrap even when the initialization "
            "segment is detected as stationary");
DEFINE_bool(write_point_clouds,
            true,
            "Write map.las and lidar_undist.dat; disable for trajectory-only diagnostics");

namespace fs = std::filesystem;

namespace {

Eigen::Matrix4d TransformFromProto(
    const reality_capture::calibration::SpatiotemporalTransform& transform) {
  const auto& q_msg = transform.rotation();
  Eigen::Quaterniond q(q_msg.w(), q_msg.x(), q_msg.y(), q_msg.z());
  if (q.norm() == 0.0) {
    q = Eigen::Quaterniond::Identity();
  } else {
    q.normalize();
  }

  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 3>(0, 0) = q.toRotationMatrix();
  result.block<3, 1>(0, 3) = Eigen::Vector3d(
      transform.translation().x(),
      transform.translation().y(),
      transform.translation().z());
  return result;
}

Eigen::Matrix3d Matrix3FromProto(
    const reality_capture::calibration::Matrix3& matrix) {
  if (matrix.row_major_values_size() != 9) {
    return Eigen::Matrix3d::Identity();
  }
  Eigen::Matrix3d result;
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      result(row, col) = matrix.row_major_values(row * 3 + col);
    }
  }
  return result;
}

Eigen::Vector3d Vector3FromProto(
    const reality_capture::calibration::Vector3& vector) {
  return {vector.x(), vector.y(), vector.z()};
}

template <typename Scalar>
Eigen::Matrix<Scalar, 3, 1> ReadVector3(const YAML::Node& node) {
  if (!node || !node.IsSequence() || node.size() != 3) {
    throw std::runtime_error("Expected a three-element YAML vector");
  }
  return {node[0].as<Scalar>(), node[1].as<Scalar>(), node[2].as<Scalar>()};
}

template <typename T>
T ReadRequired(const YAML::Node& parent,
               const std::string& section,
               const std::string& key) {
  if (!parent || !parent.IsMap()) {
    throw std::runtime_error("Missing required config section: " + section);
  }
  const YAML::Node value = parent[key];
  if (!value) {
    throw std::runtime_error("Missing required config: " + section + "." + key);
  }
  try {
    return value.as<T>();
  } catch (const YAML::Exception& error) {
    throw std::runtime_error("Invalid config " + section + "." + key +
                             ": " + error.what());
  }
}

lixel::LioParameters MakeParameters(const fs::path& calibration_path,
                                    const fs::path& config_path) {
  reality_capture::calibration::SensorCalibration calibration;
  if (!ReadSensorCalibrationFile(calibration_path.string(), calibration)) {
    throw std::runtime_error("Failed to read calibration: " +
                             calibration_path.string());
  }

  const YAML::Node config = YAML::LoadFile(config_path.string());
  lixel::LioParameters params;

  const auto preprocess = config["preprocess"];
  params.preprocess_param.range_min = preprocess["range_min"].as<double>();
  params.preprocess_param.range_max = preprocess["range_max"].as<double>();
  params.preprocess_param.body_mask_min =
      ReadVector3<double>(preprocess["body_mask_min"]);
  params.preprocess_param.body_mask_max =
      ReadVector3<double>(preprocess["body_mask_max"]);
  params.preprocess_param.sweep_duration =
      preprocess["sweep_duration"].as<double>();
  params.preprocess_param.sweep_cut_auto =
      preprocess["sweep_cut_auto"].as<bool>();
  if (const auto value = preprocess["imu_clock_drift_ppm"]) {
    params.sensor_param.imu_param.clock_drift_ppm = value.as<double>();
  }

  const auto downsample = config["downsample"];
  params.downsample_param.area_method = static_cast<lixel::SufaceAreaMethod>(
      downsample["area_method"].as<int>());
  params.downsample_param.init_pca_downsample_dis =
      downsample["init_pca_downsample_dis"].as<float>();
  params.downsample_param.base_downsample_dis =
      downsample["base_downsample_dis"].as<float>();
  params.downsample_param.max_downsample_dis =
      downsample["max_downsample_dis"].as<float>();
  params.downsample_param.ref_downsample_point_num =
      downsample["ref_downsample_point_num"].as<uint32_t>();

  const auto init = config["init_param"];
  params.init_param.init_pos_std = init["init_pos_std"].as<double>();
  params.init_param.init_vel_std = init["init_vel_std"].as<double>();
  params.init_param.init_rot_std = init["init_rot_std"].as<double>();
  params.init_param.init_acc_bias_std =
      init["init_acc_bias_std"].as<double>();
  params.init_param.init_gyro_bias_std =
      init["init_gyro_bias_std"].as<double>();
  if (const auto value = init["dynamic_init_enabled"]) {
    params.init_param.dynamic_init_enabled = value.as<bool>();
  }
  if (const auto value = init["dynamic_init_min_duration"]) {
    params.init_param.dynamic_init_min_duration = value.as<double>();
  }
  if (const auto value = init["dynamic_init_max_duration"]) {
    params.init_param.dynamic_init_max_duration = value.as<double>();
  }
  if (const auto value = init["dynamic_init_icp_voxel_size"]) {
    params.init_param.dynamic_init_icp_voxel_size = value.as<double>();
  }
  if (const auto value = init["dynamic_init_icp_max_correspondence"]) {
    params.init_param.dynamic_init_icp_max_correspondence = value.as<double>();
  }
  if (const auto value = init["dynamic_init_icp_fitness_threshold"]) {
    params.init_param.dynamic_init_icp_fitness_threshold = value.as<double>();
  }
  if (const auto value = init["dynamic_init_min_registration_points"]) {
    params.init_param.dynamic_init_min_registration_points = value.as<int>();
  }
  if (const auto value = init["dynamic_init_min_registration_ratio"]) {
    params.init_param.dynamic_init_min_registration_ratio = value.as<double>();
  }
  if (const auto value = init["dynamic_init_max_acc_std"]) {
    params.init_param.dynamic_init_max_acc_std = value.as<double>();
  }
  if (const auto value = init["dynamic_init_max_gyr_std"]) {
    params.init_param.dynamic_init_max_gyr_std = value.as<double>();
  }
  if (const auto value = init["dynamic_init_max_mean_gyr"]) {
    params.init_param.dynamic_init_max_mean_gyr = value.as<double>();
  }
  if (const auto value = init["dynamic_init_max_acc_norm_error"]) {
    params.init_param.dynamic_init_max_acc_norm_error = value.as<double>();
  }
  if (const auto value = init["dynamic_init_robust_sample_ratio"]) {
    params.init_param.dynamic_init_robust_sample_ratio = value.as<double>();
  }
  if (const auto use_initial_pose = init["use_initial_pose"]) {
    params.init_param.use_initial_pose = use_initial_pose.as<bool>();
  }
  if (params.init_param.use_initial_pose) {
    const auto q = init["initial_quaternion_xyzw"];
    if (!q || !q.IsSequence() || q.size() != 4) {
      throw std::runtime_error(
          "init_param.initial_quaternion_xyzw must contain four values");
    }
    params.init_param.initial_quaternion_xyzw = {
        q[0].as<double>(), q[1].as<double>(),
        q[2].as<double>(), q[3].as<double>()};
    params.init_param.initial_velocity =
        ReadVector3<double>(init["initial_velocity"]);
  }
  if (params.init_param.dynamic_init_min_duration < 0.0 ||
      params.init_param.dynamic_init_max_duration <
          params.init_param.dynamic_init_min_duration ||
      params.init_param.dynamic_init_icp_voxel_size <= 0.0 ||
      params.init_param.dynamic_init_icp_max_correspondence <= 0.0 ||
      params.init_param.dynamic_init_min_registration_points < 20 ||
      params.init_param.dynamic_init_min_registration_ratio <= 0.0 ||
      params.init_param.dynamic_init_min_registration_ratio > 1.0 ||
      params.init_param.dynamic_init_robust_sample_ratio <= 0.0 ||
      params.init_param.dynamic_init_robust_sample_ratio > 1.0) {
    throw std::runtime_error("Invalid dynamic initialization parameters");
  }

  const auto kf = config["kf"];
  params.kf_param.window_size = ReadRequired<int>(kf, "kf", "window_size");
  params.kf_param.reset_window_size =
      ReadRequired<int>(kf, "kf", "reset_window_size");
  params.kf_param.max_iter = ReadRequired<int>(kf, "kf", "max_iter");
  params.kf_param.acc_std = ReadRequired<double>(kf, "kf", "acc_std");
  params.kf_param.gyr_std = ReadRequired<double>(kf, "kf", "gyr_std");
  params.kf_param.acc_std_slope =
      ReadRequired<double>(kf, "kf", "acc_std_slope");
  params.kf_param.gyro_std_slope =
      ReadRequired<double>(kf, "kf", "gyro_std_slope");
  params.kf_param.acc_bias_std =
      ReadRequired<double>(kf, "kf", "acc_bias_std");
  params.kf_param.gyr_bias_std =
      ReadRequired<double>(kf, "kf", "gyr_bias_std");
  params.kf_param.faster_model =
      ReadRequired<bool>(kf, "kf", "faster_model");
  params.kf_param.knn_search_slope =
      ReadRequired<double>(kf, "kf", "knn_search_slope");
  params.kf_param.knn_search_min_dist =
      ReadRequired<double>(kf, "kf", "knn_search_min_dist");
  params.kf_param.lidar_std_dev_limit =
      ReadRequired<double>(kf, "kf", "lidar_std_dev_limit");
  params.kf_param.use_gnss = ReadRequired<bool>(kf, "kf", "use_gnss");
  params.kf_param.use_vio = ReadRequired<bool>(kf, "kf", "use_vio");
  params.kf_param.use_edge = ReadRequired<bool>(kf, "kf", "use_edge");
  if (params.kf_param.window_size < 1 ||
      params.kf_param.reset_window_size < 1 ||
      params.kf_param.max_iter < 1 ||
      params.kf_param.acc_std < 0.0 ||
      params.kf_param.gyr_std < 0.0 ||
      params.kf_param.acc_std_slope < 0.0 ||
      params.kf_param.gyro_std_slope < 0.0 ||
      params.kf_param.acc_bias_std < 0.0 ||
      params.kf_param.gyr_bias_std < 0.0 ||
      params.kf_param.knn_search_slope < 0.0 ||
      params.kf_param.knn_search_min_dist < 0.0 ||
      params.kf_param.lidar_std_dev_limit <= 0.0) {
    throw std::runtime_error("Invalid kf configuration values");
  }

  params.map_param.map_type = lixel::MapType::XMAP;
  params.map_param.config_path = config_path.string();

  if (calibration.has_imu_intrinsics() &&
      calibration.imu_intrinsics().has_tpm_icra_2014()) {
    const auto& imu = calibration.imu_intrinsics().tpm_icra_2014();
    auto& out = params.sensor_param.imu_param;
    out.Ta = Matrix3FromProto(imu.accelerometer_ta());
    out.Ka = Matrix3FromProto(imu.accelerometer_ka());
    out.Ba = Vector3FromProto(imu.accelerometer_bias());
    out.Tg = Matrix3FromProto(imu.gyroscope_tg());
    out.Kg = Matrix3FromProto(imu.gyroscope_kg());
    out.Bg = Vector3FromProto(imu.gyroscope_bias());
    out.calibrated = true;
    out.enabled = true;
  } else {
    auto& out = params.sensor_param.imu_param;
    out.Ta = Eigen::Matrix3d::Identity();
    out.Ka = Eigen::Matrix3d::Identity();
    out.Ba = Eigen::Vector3d::Zero();
    out.Tg = Eigen::Matrix3d::Identity();
    out.Kg = Eigen::Matrix3d::Identity();
    out.Bg = Eigen::Vector3d::Zero();
    out.enabled = false;
  }

  auto& lidar = params.sensor_param.lidar_param;
  if (calibration.has_lidar_intrinsics() &&
      calibration.lidar_intrinsics().has_simple()) {
    lidar.elevation_offset = calibration.lidar_intrinsics().simple().e();
    lidar.calibrated = true;
    lidar.enabled = true;
  } else {
    lidar.enabled = false;
  }

  params.extrinsic_param.motor_param.enabled = false;
  if (calibration.has_direct()) {
    const auto& imu_from_lidar = calibration.direct().imu_from_lidar();
    params.extrinsic_param.ext_imu_motor =
        TransformFromProto(imu_from_lidar);
    params.sensor_param.imu_param.lidar_to_imu_time_offset_seconds =
        imu_from_lidar.time_offset_seconds();
  } else if (calibration.has_via_encoder()) {
    const auto& chain = calibration.via_encoder();
    params.extrinsic_param.motor_param.ext_motor_lidar =
        TransformFromProto(chain.encoder_from_lidar());
    params.extrinsic_param.ext_imu_motor =
        TransformFromProto(chain.imu_from_encoder());
    params.sensor_param.imu_param.lidar_to_imu_time_offset_seconds =
        chain.encoder_from_lidar().time_offset_seconds() +
        chain.imu_from_encoder().time_offset_seconds();
    params.extrinsic_param.motor_param.enabled = true;
  } else {
    throw std::runtime_error("Calibration has no LiDAR extrinsics");
  }

  return params;
}

class ResultWriter {
 public:
  explicit ResultWriter(const fs::path& output_dir, bool write_point_clouds)
      : output_dir_(output_dir), write_point_clouds_(write_point_clouds) {
    fs::create_directories(output_dir_);

    trajectory_.open(output_dir_ / "trajectory.txt");
    if (!trajectory_) {
      throw std::runtime_error("Failed to create trajectory.txt");
    }
    trajectory_ << "#x y z roll pitch yaw qx qy qz qw timestamp\n";

    state_diagnostics_.open(output_dir_ / "state_diagnostics.txt");
    if (!state_diagnostics_) {
      throw std::runtime_error("Failed to create state_diagnostics.txt");
    }
    state_diagnostics_
        << "#timestamp px py pz vx vy vz bax bay baz bgx bgy bgz qx qy qz qw\n";

    frame_diagnostics_.open(output_dir_ / "frame_diagnostics.txt");
    if (!frame_diagnostics_) {
      throw std::runtime_error("Failed to create frame_diagnostics.txt");
    }
    frame_diagnostics_
        << "#state_timestamp measurement_timestamp sweep_id update_ok downsample matched_current "
           "matched_total input_total search_ok plane_ok overlap "
           "normal_assigned_ratio "
           "residual_mean residual_rms residual_std "
           "info_eig0 info_eig1 info_eig2 info_eig3 info_eig4 info_eig5 "
           "iter pos_dx pos_dy pos_dz rot_dx rot_dy rot_dz "
           "vel_dx vel_dy vel_dz pos_sx pos_sy pos_sz "
           "rot_sx rot_sy rot_sz vel_sx vel_sy vel_sz\n";

    ulog_storage_ = std::make_unique<middleware::ULogStorage>();
    ulog_storage_->StartLog((output_dir_ / "lio.ulg").string());

    if (write_point_clouds_) {
      point_table_.layout()->registerDim(pdal::Dimension::Id::X);
      point_table_.layout()->registerDim(pdal::Dimension::Id::Y);
      point_table_.layout()->registerDim(pdal::Dimension::Id::Z);
      point_table_.layout()->registerDim(pdal::Dimension::Id::GpsTime);
      point_table_.layout()->registerDim(pdal::Dimension::Id::Intensity);
      las_writer_.initialize((output_dir_ / "map.las").string(), point_table_);

      if (!lidar_writer_.Open((output_dir_ / "lidar_undist.dat").string())) {
        throw std::runtime_error("Failed to create lidar_undist.dat");
      }
    }
  }

  ResultWriter(const ResultWriter&) = delete;
  ResultWriter& operator=(const ResultWriter&) = delete;

  ~ResultWriter() {
    try {
      Finalize();
    } catch (const std::exception& error) {
      spdlog::error("Failed to finalize realtime outputs: {}", error.what());
    }
  }

  void Write(const lixel::LioResultMsg::Ptr& result) {
    if (!result || !result->body_points || result->body_points->empty()) {
      return;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    const double timestamp = result->full_state.timestamp;
    const Eigen::Quaterniond q = result->full_state.q.normalized();
    const Eigen::Vector3d p = result->full_state.p;

    std::shared_ptr<proto::LidarMsg> lidar;
    pdal::PointViewPtr view;
    if (write_point_clouds_) {
      lidar = std::make_shared<proto::LidarMsg>();
      lidar->mutable_points()->Reserve(
          static_cast<int>(result->body_points->size()));
      view.reset(new pdal::PointView(point_table_));
    }

    for (size_t index = 0; index < result->body_points->size(); ++index) {
      const auto& point = result->body_points->points[index];

      if (!write_point_clouds_) {
        continue;
      }

      auto* output_point = lidar->add_points();
      output_point->set_timestamp(timestamp);
      output_point->set_x(point.x);
      output_point->set_y(point.y);
      output_point->set_z(point.z);
      output_point->set_intensity(
          static_cast<uint32_t>(std::clamp(point.intensity, 0.0f, 65535.0f)));

      const Eigen::Vector3d world =
          q * Eigen::Vector3d(point.x, point.y, point.z) + p;
      view->setField(pdal::Dimension::Id::X, index, world.x());
      view->setField(pdal::Dimension::Id::Y, index, world.y());
      view->setField(pdal::Dimension::Id::Z, index, world.z());
      view->setField(pdal::Dimension::Id::GpsTime, index, timestamp);
      view->setField(
          pdal::Dimension::Id::Intensity,
          index,
          static_cast<uint16_t>(std::clamp(point.intensity, 0.0f, 65535.0f)));
    }

    if (write_point_clouds_) {
      lidar_writer_.Write(lidar);
      las_writer_.writeView(view);
    }

    auto* pose = poses_.add_pose_msgs();
    pose->set_timestamp(timestamp);
    pose->set_tx(p.x());
    pose->set_ty(p.y());
    pose->set_tz(p.z());
    pose->set_rx(q.x());
    pose->set_ry(q.y());
    pose->set_rz(q.z());
    pose->set_rw(q.w());
    pose->set_gx(result->full_state.gravity.x());
    pose->set_gy(result->full_state.gravity.y());
    pose->set_gz(result->full_state.gravity.z());

    const Eigen::Vector3d rpy = q.toRotationMatrix().eulerAngles(0, 1, 2);
    ulog_storage_->HandleFullState(timestamp, result->full_state);
    ulog_storage_->HandleIeskfAttribute(
        timestamp, result->attribute_ieskf);
    ulog_storage_->HandleIeskfStatePredict(
        result->attribute_ieskf.sweep_id,
        result->attribute_ieskf.state_predict);
    ulog_storage_->HandleIeskfAttributePredict(
        result->attribute_ieskf.sweep_id,
        result->attribute_ieskf.attritube_predict);
    ulog_storage_->HandleDebugMsgs(result->debug_msgs);
    trajectory_ << std::fixed << std::setprecision(12)
                << p.x() << ' ' << p.y() << ' ' << p.z() << ' '
                << rpy.x() << ' ' << rpy.y() << ' ' << rpy.z() << ' '
                << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
                << ' ' << timestamp << '\n';
    state_diagnostics_ << std::fixed << std::setprecision(12)
                       << timestamp << ' '
                       << p.x() << ' ' << p.y() << ' ' << p.z() << ' '
                       << result->full_state.v.x() << ' '
                       << result->full_state.v.y() << ' '
                       << result->full_state.v.z() << ' '
                       << result->full_state.ba.x() << ' '
                       << result->full_state.ba.y() << ' '
                       << result->full_state.ba.z() << ' '
                       << result->full_state.bg.x() << ' '
                       << result->full_state.bg.y() << ' '
                       << result->full_state.bg.z() << ' '
                       << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
                       << '\n';
    const auto& attribute = result->attribute_ieskf;
    const auto& jacobi = attribute.jacobi;
    const auto& iterate = attribute.iterate;
    frame_diagnostics_ << std::fixed << std::setprecision(12)
                       << timestamp << ' ' << attribute.timestamp << ' '
                       << attribute.sweep_id << ' '
                       << static_cast<int>(attribute.update_success) << ' '
                       << attribute.downsample_dis << ' '
                       << jacobi.current_use_point_num << ' '
                       << jacobi.use_point_num << ' '
                       << jacobi.total_point_num << ' '
                       << jacobi.search_success_num << ' '
                       << jacobi.plane_success_num << ' '
                       << jacobi.overlap_radio << ' '
                       << jacobi.normal_assigned_ratio << ' '
                       << jacobi.residual_mean << ' '
                       << jacobi.residual_rms << ' '
                       << iterate.std_dev << ' ';
    for (int index = 0; index < 6; ++index) {
      frame_diagnostics_ << jacobi.current_pose_information_eig(index) << ' ';
    }
    frame_diagnostics_ << iterate.iter_num << ' '
                       << iterate.pos_update.x() << ' '
                       << iterate.pos_update.y() << ' '
                       << iterate.pos_update.z() << ' '
                       << iterate.rot_update.x() << ' '
                       << iterate.rot_update.y() << ' '
                       << iterate.rot_update.z() << ' '
                       << iterate.vel_update.x() << ' '
                       << iterate.vel_update.y() << ' '
                       << iterate.vel_update.z() << ' '
                       << iterate.pos_std.x() << ' '
                       << iterate.pos_std.y() << ' '
                       << iterate.pos_std.z() << ' '
                       << iterate.rot_std.x() << ' '
                       << iterate.rot_std.y() << ' '
                       << iterate.rot_std.z() << ' '
                       << iterate.vel_std.x() << ' '
                       << iterate.vel_std.y() << ' '
                       << iterate.vel_std.z() << '\n';

    ++frame_count_;
    if (frame_count_ % 100 == 0) {
      spdlog::info("Saved {} realtime body-frame scans", frame_count_);
    }
  }

  void Finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    if (finalized_) {
      return;
    }
    finalized_ = true;
    if (write_point_clouds_) {
      lidar_writer_.Close();
    }
    trajectory_.flush();
    trajectory_.close();
    state_diagnostics_.flush();
    state_diagnostics_.close();
    frame_diagnostics_.flush();
    frame_diagnostics_.close();
    if (ulog_storage_) {
      ulog_storage_->StopLog();
    }
    if (write_point_clouds_) {
      las_writer_.finalize(point_table_);
    }
    if (!WritePoseFile((output_dir_ / "traj.dat").string(), poses_)) {
      throw std::runtime_error("Failed to write traj.dat");
    }
    spdlog::info(
        "Finalized {} synchronized body poses and undistorted scans",
        frame_count_);
  }

 private:
  fs::path output_dir_;
  std::mutex mutex_;
  std::ofstream trajectory_;
  std::ofstream state_diagnostics_;
  std::ofstream frame_diagnostics_;
  std::unique_ptr<middleware::ULogStorage> ulog_storage_;
  pdal::PointTable point_table_;
  migration::IncrementalLasWriter las_writer_;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer_;
  proto::PoseMsgList poses_;
  size_t frame_count_ = 0;
  bool finalized_ = false;
  bool write_point_clouds_ = true;
};

struct KtrjPose {
  double timestamp;
  double tx;
  double ty;
  double tz;
  double qx;
  double qy;
  double qz;
  double qw;
};
static_assert(sizeof(KtrjPose) == 64);

std::vector<KtrjPose> ReadKtrjPoses(const fs::path& path) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    throw std::runtime_error("Failed to open KTRJ trajectory: " + path.string());
  }
  char magic[4] = {};
  uint32_t version = 0;
  uint32_t reserved = 0;
  input.read(magic, sizeof(magic));
  input.read(reinterpret_cast<char*>(&version), sizeof(version));
  input.read(reinterpret_cast<char*>(&reserved), sizeof(reserved));
  if (std::string(magic, sizeof(magic)) != "KTRJ" || version != 1) {
    throw std::runtime_error("Unsupported KTRJ header: " + path.string());
  }
  std::vector<KtrjPose> poses;
  KtrjPose pose{};
  while (input.read(reinterpret_cast<char*>(&pose), sizeof(pose))) {
    poses.push_back(pose);
  }
  if (!input.eof() || poses.size() < 2) {
    throw std::runtime_error("Invalid or empty KTRJ trajectory: " + path.string());
  }
  return poses;
}

std::pair<Eigen::Vector3d, Eigen::Quaterniond> InterpolateKtrj(
    const std::vector<KtrjPose>& poses, double timestamp) {
  const auto upper = std::lower_bound(
      poses.begin(), poses.end(), timestamp,
      [](const KtrjPose& pose, double value) {
        return pose.timestamp < value;
      });
  if (upper == poses.begin()) {
    const auto& pose = poses.front();
    return {{pose.tx, pose.ty, pose.tz},
            Eigen::Quaterniond(pose.qw, pose.qx, pose.qy, pose.qz).normalized()};
  }
  if (upper == poses.end()) {
    const auto& pose = poses.back();
    return {{pose.tx, pose.ty, pose.tz},
            Eigen::Quaterniond(pose.qw, pose.qx, pose.qy, pose.qz).normalized()};
  }
  const auto& before = *(upper - 1);
  const auto& after = *upper;
  const double alpha = std::clamp(
      (timestamp - before.timestamp) / (after.timestamp - before.timestamp),
      0.0, 1.0);
  const Eigen::Vector3d p0(before.tx, before.ty, before.tz);
  const Eigen::Vector3d p1(after.tx, after.ty, after.tz);
  const Eigen::Quaterniond q0(before.qw, before.qx, before.qy, before.qz);
  const Eigen::Quaterniond q1(after.qw, after.qx, after.qy, after.qz);
  return {(1.0 - alpha) * p0 + alpha * p1,
          q0.normalized().slerp(alpha, q1.normalized()).normalized()};
}

void ReplayExternalTrajectory(
    const fs::path& input_dir,
    const fs::path& trajectory_path,
    const Eigen::Matrix4d& imu_from_lidar,
    ResultWriter* writer) {
  const auto poses = ReadKtrjPoses(trajectory_path);
  const Eigen::Matrix3d imu_from_lidar_rotation =
      imu_from_lidar.block<3, 3>(0, 0);
  const Eigen::Vector3d imu_from_lidar_translation =
      imu_from_lidar.block<3, 1>(0, 3);

  SequentialLidarFileReader<proto::LidarMsg> lidar_reader;
  if (!lidar_reader.Open((input_dir / "lidar.dat").string())) {
    throw std::runtime_error("Failed to open lidar.dat");
  }

  size_t frame_count = 0;
  std::shared_ptr<proto::LidarMsg> scan;
  while (lidar_reader.ReadNext(scan)) {
    if (!scan || scan->points().empty()) {
      continue;
    }
    const double scan_timestamp = scan->points().rbegin()->timestamp();
    const auto [scan_position, scan_rotation] =
        InterpolateKtrj(poses, scan_timestamp);
    lixel::PointCloud::Ptr body_points(new lixel::PointCloud);
    body_points->reserve(scan->points_size());
    for (const auto& point : scan->points()) {
      const auto [point_position, point_rotation] =
          InterpolateKtrj(poses, point.timestamp());
      const Eigen::Vector3d point_lidar(point.x(), point.y(), point.z());
      const Eigen::Vector3d point_imu =
          imu_from_lidar_rotation * point_lidar + imu_from_lidar_translation;
      const Eigen::Vector3d point_world =
          point_rotation * point_imu + point_position;
      const Eigen::Vector3d point_body =
          scan_rotation.conjugate() * (point_world - scan_position);
      lixel::PointIRT output;
      output.x = static_cast<float>(point_body.x());
      output.y = static_cast<float>(point_body.y());
      output.z = static_cast<float>(point_body.z());
      output.intensity = static_cast<float>(point.intensity());
      output.timestamp = point.timestamp();
      output.ring = 0;
      body_points->push_back(output);
    }
    body_points->width = static_cast<uint32_t>(body_points->size());
    body_points->height = 1;
    auto result = std::make_shared<lixel::LioResultMsg>();
    result->full_state = lixel::FullStateMsg(
        scan_timestamp, scan_rotation, scan_position,
        Eigen::Vector3d::Zero(), Eigen::Vector3d::Zero(),
        Eigen::Vector3d::Zero(), Eigen::Vector3d(0.0, 0.0, -9.8));
    result->body_points = body_points;
    writer->Write(result);
    ++frame_count;
    if (frame_count % 100 == 0) {
      spdlog::info("Replayed {} KTRJ-guided scans", frame_count);
    }
  }
  lidar_reader.Close();
}

fs::path ExecutableDirectory() {
#ifdef _WIN32
  std::wstring buffer(32768, L'\0');
  const DWORD size = GetModuleFileNameW(
      nullptr, buffer.data(), static_cast<DWORD>(buffer.size()));
  if (size == 0 || size == buffer.size()) {
    throw std::runtime_error("Failed to determine executable directory");
  }
  buffer.resize(size);
  return fs::path(buffer).parent_path();
#else
  return fs::current_path();
#endif
}

}  // namespace

int main(int argc, char** argv) {
  google::InitGoogleLogging(argv[0]);
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  try {
    if (FLAGS_input_dir.empty() || FLAGS_output_dir.empty()) {
      throw std::runtime_error("--input_dir and --output_dir are required");
    }

    const fs::path input_dir = fs::absolute(FLAGS_input_dir);
    const fs::path output_dir = fs::absolute(FLAGS_output_dir);
    fs::path config_path(FLAGS_config_filename);
    if (config_path.is_relative()) {
      config_path = ExecutableDirectory() / config_path;
    }
    lixel::LioParameters parameters =
        MakeParameters(input_dir / "calibration.dat", config_path);
    std::cout << "[PHOENIX_TIME] lidar_to_imu_offset="
              << parameters.sensor_param.imu_param.lidar_to_imu_time_offset_seconds
              << " imu_clock_drift_ppm="
              << parameters.sensor_param.imu_param.clock_drift_ppm << std::endl;
    if (FLAGS_force_dynamic_initialization) {
      // This is a diagnostic switch for exercising the complete moving-start
      // handoff on recordings whose first seconds happen to be stationary.
      // Negative stationarity limits make the decision deterministic without
      // changing the production YAML or the registration-quality checks.
      parameters.init_param.dynamic_init_max_acc_std = -1.0;
      parameters.init_param.dynamic_init_max_gyr_std = -1.0;
      parameters.init_param.dynamic_init_max_mean_gyr = -1.0;
      parameters.init_param.dynamic_init_max_acc_norm_error = -1.0;
    }

    ResultWriter writer(output_dir, FLAGS_write_point_clouds);
    if (!FLAGS_external_trajectory_bin.empty()) {
      ReplayExternalTrajectory(
          input_dir, fs::absolute(FLAGS_external_trajectory_bin),
          parameters.extrinsic_param.ext_imu_motor, &writer);
      writer.Finalize();
      return 0;
    }
    lixel::LioCore core(parameters);
    core.setLioResultCallback(
        [&writer](const lixel::LioResultMsg::Ptr& result) {
          writer.Write(result);
        });

    proto::ImuMsgList imu_messages;
    if (!ReadImuFile((input_dir / "imu.dat").string(), imu_messages)) {
      throw std::runtime_error("Failed to read imu.dat");
    }

    core.start();
    for (const auto& imu : imu_messages.imu_msgs()) {
      auto message = std::make_shared<lixel::ImuMsg>(
          imu.timestamp(),
          Eigen::Vector3d(imu.ax(), imu.ay(), imu.az()),
          Eigen::Vector3d(imu.gx(), imu.gy(), imu.gz()));
      core.addSensorData(message);
    }

    SequentialLidarFileReader<proto::LidarMsg> lidar_reader;
    if (!lidar_reader.Open((input_dir / "lidar.dat").string())) {
      throw std::runtime_error("Failed to open lidar.dat");
    }

    size_t source_frame_id = 0;
    size_t fed_frame_count = 0;
    std::shared_ptr<proto::LidarMsg> scan;
    while (lidar_reader.ReadNext(scan)) {
      const size_t current_source_frame = source_frame_id++;
      if (current_source_frame < static_cast<size_t>(FLAGS_skip_lidar_frames)) {
        continue;
      }
      if (!scan || scan->points().empty()) {
        continue;
      }
      lixel::PointCloud::Ptr cloud(new lixel::PointCloud);
      cloud->reserve(scan->points_size());
      for (const auto& point : scan->points()) {
        lixel::PointIRT output;
        output.x = point.x();
        output.y = point.y();
        output.z = point.z();
        output.intensity = static_cast<float>(point.intensity());
        output.timestamp = point.timestamp();
        output.ring = 0;
        cloud->push_back(output);
      }
      cloud->width = static_cast<uint32_t>(cloud->size());
      cloud->height = 1;

      auto message = std::make_shared<lixel::PointCloudMsg>(
          cloud->back().timestamp, cloud);
      message->frame_id = static_cast<int>(current_source_frame);
      core.addSensorData(message);
      ++fed_frame_count;

      // Keep the offline producer close to the mapping consumer. Without
      // backpressure, a long recording is decoded into the in-memory LiDAR
      // queue much faster than mapping can consume it, eventually exhausting
      // the process commit limit. Leave only the short synchronization window
      // required by IOUtils buffered; the sensor ordering and SLAM inputs stay
      // unchanged.
      while (core.containsEnoughDataForSyncPackages()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
      }
      if (fed_frame_count == 1) {
        std::cout << "[PHOENIX] segment_start source_frame="
                  << current_source_frame << " timestamp="
                  << std::fixed << std::setprecision(6)
                  << cloud->back().timestamp << std::endl;
      }
      if (fed_frame_count % 100 == 0) {
        std::cout << "[PHOENIX] fed_frames=" << fed_frame_count
                  << " source_frame=" << current_source_frame
                  << " input_progress=" << lidar_reader.getProgress() << "%" << std::endl;
      }
      if (FLAGS_max_lidar_frames > 0 &&
          fed_frame_count >= static_cast<size_t>(FLAGS_max_lidar_frames)) {
        std::cout << "[PHOENIX] reached max_lidar_frames=" << FLAGS_max_lidar_frames << std::endl;
        break;
      }
    }
    lidar_reader.Close();

    while (core.containsEnoughDataForSyncPackages()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }
    core.stop();
    writer.Finalize();
  } catch (const std::exception& error) {
    spdlog::error("Realtime mapping failed: {}", error.what());
    return 1;
  }

  return 0;
}
