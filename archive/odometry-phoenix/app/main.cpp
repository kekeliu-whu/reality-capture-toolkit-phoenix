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
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <thread>

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

  const auto kf = config["kf"];
  params.kf_param.acc_std = kf["acc_std"].as<double>();
  params.kf_param.gyr_std = kf["gyr_std"].as<double>();
  params.kf_param.acc_bias_std = kf["acc_bias_std"].as<double>();
  params.kf_param.gyr_bias_std = kf["gyr_bias_std"].as<double>();
  params.kf_param.max_iter = kf["max_iter"].as<int>();
  params.kf_param.acc_keep_std_limit =
      kf["acc_keep_std_limit"].as<double>();
  params.kf_param.gyro_keep_std_limit =
      kf["gyro_keep_std_limit"].as<double>();
  params.kf_param.use_gnss = kf["use_gnss"].as<bool>();
  params.kf_param.use_vio = kf["use_vio"].as<bool>();
  params.kf_param.use_edge = kf["use_edge"].as<bool>();
  params.kf_param.k_for_adaptive_search =
      kf["k_for_adaptive_search"].as<double>();

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
    params.extrinsic_param.ext_imu_motor =
        TransformFromProto(calibration.direct().imu_from_lidar());
  } else if (calibration.has_via_encoder()) {
    params.extrinsic_param.motor_param.ext_motor_lidar =
        TransformFromProto(calibration.via_encoder().encoder_from_lidar());
    params.extrinsic_param.ext_imu_motor =
        TransformFromProto(calibration.via_encoder().imu_from_encoder());
    params.extrinsic_param.motor_param.enabled = true;
  } else {
    throw std::runtime_error("Calibration has no LiDAR extrinsics");
  }

  return params;
}

class ResultWriter {
 public:
  explicit ResultWriter(const fs::path& output_dir)
      : output_dir_(output_dir) {
    fs::create_directories(output_dir_);

    trajectory_.open(output_dir_ / "trajectory.txt");
    if (!trajectory_) {
      throw std::runtime_error("Failed to create trajectory.txt");
    }
    trajectory_ << "#x y z roll pitch yaw qx qy qz qw timestamp\n";

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

    auto lidar = std::make_shared<proto::LidarMsg>();
    lidar->mutable_points()->Reserve(
        static_cast<int>(result->body_points->size()));
    pdal::PointViewPtr view(new pdal::PointView(point_table_));

    for (size_t index = 0; index < result->body_points->size(); ++index) {
      const auto& point = result->body_points->points[index];

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

    lidar_writer_.Write(lidar);
    las_writer_.writeView(view);

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
    trajectory_ << std::fixed << std::setprecision(12)
                << p.x() << ' ' << p.y() << ' ' << p.z() << ' '
                << rpy.x() << ' ' << rpy.y() << ' ' << rpy.z() << ' '
                << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w()
                << ' ' << timestamp << '\n';

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
    lidar_writer_.Close();
    trajectory_.flush();
    trajectory_.close();
    las_writer_.finalize(point_table_);
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
  pdal::PointTable point_table_;
  migration::IncrementalLasWriter las_writer_;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer_;
  proto::PoseMsgList poses_;
  size_t frame_count_ = 0;
  bool finalized_ = false;
};

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

    ResultWriter writer(output_dir);
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

    size_t frame_id = 0;
    std::shared_ptr<proto::LidarMsg> scan;
    while (lidar_reader.ReadNext(scan)) {
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
      message->frame_id = static_cast<int>(frame_id++);
      core.addSensorData(message);

      // Keep the offline producer close to the mapping consumer. Without
      // backpressure, a long recording is decoded into the in-memory LiDAR
      // queue much faster than mapping can consume it, eventually exhausting
      // the process commit limit. Leave only the short synchronization window
      // required by IOUtils buffered; the sensor ordering and SLAM inputs stay
      // unchanged.
      while (core.containsEnoughDataForSyncPackages()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(2));
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
