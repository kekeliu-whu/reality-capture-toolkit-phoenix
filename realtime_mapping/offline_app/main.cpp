#include <gflags/gflags.h>
#include <migration/inc_las_writer.h>
#include <migration/proto_io.h>
#include <pdal/Dimension.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <spdlog/spdlog.h>

#include <atomic>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <memory>
#include <mutex>
#include <thread>

#include "lio.h"

DEFINE_string(input_dir, "", "Directory containing calibration.dat, imu.dat and lidar.dat");
DEFINE_string(output_dir, "", "Directory for realtime-mapping results");

namespace fs = std::filesystem;

namespace {

Eigen::Matrix4d TransformFromProto(
    const reality_capture::calibration::SpatiotemporalTransform& transform) {
  const auto& q = transform.rotation();
  Eigen::Quaterniond rotation(q.w(), q.x(), q.y(), q.z());
  Eigen::Matrix4d result = Eigen::Matrix4d::Identity();
  result.block<3, 3>(0, 0) = rotation.normalized().toRotationMatrix();
  result(0, 3) = transform.translation().x();
  result(1, 3) = transform.translation().y();
  result(2, 3) = transform.translation().z();
  return result;
}

lixel::LioParameters MakeParameters(const fs::path& config_path,
                                    const fs::path& calibration_path) {
  lixel::LioParameters params{};

  reality_capture::calibration::SensorCalibration calibration;
  if (!ReadSensorCalibrationFile(calibration_path.string(), calibration)) {
    throw std::runtime_error("failed to read " + calibration_path.string());
  }
  if (!calibration.has_direct()) {
    throw std::runtime_error("realtime_mapping offline app requires direct LiDAR-to-IMU extrinsics");
  }
  params.extrinsic_param.ext_imu_motor =
      TransformFromProto(calibration.direct().imu_from_lidar());
  params.extrinsic_param.motor_param.enabled = false;

  // Kosmo/Hesai conversion currently supplies the rigid extrinsic only.
  params.sensor_param.imu_param.enabled = false;
  params.sensor_param.lidar_param.enabled = false;

  params.preprocess_param.range_min = 0.3;
  params.preprocess_param.range_max = 70.0;
  params.preprocess_param.body_mask_min = Eigen::Vector3d(-0.5, -0.5, -0.2);
  params.preprocess_param.body_mask_max = Eigen::Vector3d(0.0, 0.5, 0.6);
  params.preprocess_param.sweep_duration = 0.1;
  params.preprocess_param.sweep_cut_auto = false;

  params.downsample_param.area_method = lixel::SufaceAreaMethod::ellipsoid;
  params.downsample_param.init_pca_downsample_dis = 1.5f;
  params.downsample_param.base_downsample_dis = 0.2f;
  params.downsample_param.max_downsample_dis = 0.3f;
  params.downsample_param.ref_downsample_point_num = 4000;

  params.init_param.init_pos_std = 0.001;
  params.init_param.init_vel_std = 0.001;
  params.init_param.init_rot_std = 0.001;
  params.init_param.init_acc_bias_std = 0.001;
  params.init_param.init_gyro_bias_std = 0.00005;

  params.kf_param.acc_std = 0.080;
  params.kf_param.gyr_std = 0.003;
  params.kf_param.acc_bias_std = 0.0020;
  params.kf_param.gyr_bias_std = 0.0002;
  params.kf_param.max_iter = 5;
  params.kf_param.acc_keep_std_limit = 0.5;
  params.kf_param.gyro_keep_std_limit = 0.5;
  params.kf_param.use_gnss = false;
  params.kf_param.use_vio = false;
  params.kf_param.use_edge = false;
  params.kf_param.k_for_adaptive_search = 0.003;
  params.map_param.config_path = config_path.string();
  return params;
}

class ResultWriter {
 public:
  explicit ResultWriter(const fs::path& output_dir)
      : trajectory_(output_dir / "trajectory.txt"),
        las_writer_(std::make_unique<migration::IncrementalLasWriter>()) {
    if (!trajectory_) {
      throw std::runtime_error("failed to create trajectory.txt");
    }
    trajectory_ << "#x y z roll pitch yaw qx qy qz qw timestamp\n";
    table_.layout()->registerDim(pdal::Dimension::Id::X);
    table_.layout()->registerDim(pdal::Dimension::Id::Y);
    table_.layout()->registerDim(pdal::Dimension::Id::Z);
    table_.layout()->registerDim(pdal::Dimension::Id::GpsTime);
    table_.layout()->registerDim(pdal::Dimension::Id::Intensity);
    las_writer_->initialize((output_dir / "map.las").string(), table_);
  }

  void Write(const lixel::LioResultMsg::Ptr& result) {
    std::lock_guard<std::mutex> lock(mutex_);
    const Eigen::Quaterniond q(result->full_state.q);
    const Eigen::Vector3d p(result->full_state.p);
    const Eigen::Vector3d rpy = q.toRotationMatrix().eulerAngles(0, 1, 2);
    trajectory_ << std::fixed << std::setprecision(12)
                << p.x() << ' ' << p.y() << ' ' << p.z() << ' '
                << rpy.x() << ' ' << rpy.y() << ' ' << rpy.z() << ' '
                << q.x() << ' ' << q.y() << ' ' << q.z() << ' ' << q.w() << ' '
                << result->full_state.timestamp << '\n';

    pdal::PointViewPtr view(new pdal::PointView(table_));
    const Eigen::Matrix3d rotation = q.toRotationMatrix();
    for (pdal::PointId i = 0; i < result->body_points->size(); ++i) {
      const auto& point = result->body_points->points[i];
      const Eigen::Vector3d world = rotation * Eigen::Vector3d(point.x, point.y, point.z) + p;
      view->setField(pdal::Dimension::Id::X, i, world.x());
      view->setField(pdal::Dimension::Id::Y, i, world.y());
      view->setField(pdal::Dimension::Id::Z, i, world.z());
      view->setField(pdal::Dimension::Id::GpsTime, i, result->full_state.timestamp);
      view->setField(pdal::Dimension::Id::Intensity, i, static_cast<uint16_t>(point.intensity));
    }
    las_writer_->writeView(view);
    ++result_count_;
  }

  void Finalize() {
    std::lock_guard<std::mutex> lock(mutex_);
    trajectory_.flush();
    las_writer_->finalize(table_);
  }

  size_t result_count() const { return result_count_.load(); }

 private:
  std::ofstream trajectory_;
  pdal::PointTable table_;
  std::unique_ptr<migration::IncrementalLasWriter> las_writer_;
  std::mutex mutex_;
  std::atomic<size_t> result_count_{0};
};

}  // namespace

int main(int argc, char** argv) {
  std::set_terminate([]() {
    if (auto error = std::current_exception()) {
      try {
        std::rethrow_exception(error);
      } catch (const std::exception& exception) {
        std::fprintf(stderr, "Unhandled exception: %s\n", exception.what());
      } catch (...) {
        std::fprintf(stderr, "Unhandled non-standard exception\n");
      }
    } else {
      std::fprintf(stderr, "std::terminate called without an active exception\n");
    }
    std::fflush(stderr);
    std::abort();
  });
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_input_dir.empty()) {
    spdlog::error("--input_dir is required");
    return 2;
  }

  const fs::path input_dir = fs::absolute(FLAGS_input_dir);
  const fs::path output_dir = FLAGS_output_dir.empty()
                                  ? input_dir / "realtimemapping_w5"
                                  : fs::absolute(FLAGS_output_dir);
  fs::create_directories(output_dir);

  const fs::path exe_dir = fs::absolute(argv[0]).parent_path();
  try {
    auto params = MakeParameters(exe_dir / "K1.yaml", input_dir / "calibration.dat");
    proto::ImuMsgList imu_data;
    if (!ReadImuFile((input_dir / "imu.dat").string(), imu_data)) {
      throw std::runtime_error("failed to read imu.dat");
    }
    spdlog::info("Loaded {} IMU samples", imu_data.imu_msgs_size());

    ResultWriter writer(output_dir);
    lixel::LioCore lio(params);
    lio.setLioResultCallback([&writer](const lixel::LioResultMsg::Ptr& result) {
      writer.Write(result);
    });
    for (const auto& imu : imu_data.imu_msgs()) {
      auto msg = std::make_shared<lixel::ImuMsg>();
      msg->timestamp = imu.timestamp();
      msg->acc = Eigen::Vector3d(imu.ax(), imu.ay(), imu.az());
      msg->gyro = Eigen::Vector3d(imu.gx(), imu.gy(), imu.gz());
      lio.addSensorData(msg);
    }

    SequentialLidarFileReader<proto::LidarMsg> lidar_reader;
    if (!lidar_reader.Open((input_dir / "lidar.dat").string())) {
      throw std::runtime_error("failed to read lidar.dat");
    }
    int frame_id = 0;
    bool lio_started = false;
    while (!lidar_reader.IsFileEnded()) {
      Ptr<proto::LidarMsg> scan;
      if (frame_id < 3) spdlog::info("Reading input frame {}", frame_id);
      if (!lidar_reader.ReadNext(scan)) break;
      if (!scan || scan->points().empty()) continue;
      if (frame_id < 3) {
        spdlog::info("Frame {} has {} points, first time {:.9f}, last time {:.9f}",
                     frame_id, scan->points_size(), scan->points(0).timestamp(),
                     scan->points(scan->points_size() - 1).timestamp());
      }

      auto msg = std::make_shared<lixel::PointCloudMsg>();
      msg->frame_id = frame_id++;
      msg->timestamp = scan->points(0).timestamp();
      msg->points = std::make_shared<lixel::PointCloud>();
      msg->points->reserve(scan->points_size());
      for (const auto& source : scan->points()) {
        lixel::PointT point;
        point.x = source.x();
        point.y = source.y();
        point.z = source.z();
        point.intensity = static_cast<float>(source.intensity());
        point.timestamp = source.timestamp();
        point.ring = 0;
        msg->points->push_back(point);
      }
      if (frame_id < 3) spdlog::info("Submitting input frame {}", frame_id);
      lio.addSensorData(msg);
      if (frame_id < 3) spdlog::info("Submitted input frame {}", frame_id);

      if (!lio_started && frame_id >= 10) {
        spdlog::info("Starting LIO after preloading {} frames", frame_id);
        lio.start();
        spdlog::info("LIO started");
        lio_started = true;
      }

      if (frame_id == 10) spdlog::info("Checking buffered duration");
      while (lio_started && lio.getInputDataCommonDuration() > 1.0) {
        std::this_thread::sleep_for(std::chrono::milliseconds(20));
      }
      if (frame_id == 10) spdlog::info("Buffered duration check complete");
      if (frame_id % 100 == 0) {
        spdlog::info("Input progress {:.1f}%, frames {}, results {}",
                     lidar_reader.getProgress(), frame_id, writer.result_count());
      }
    }

    if (!lio_started) {
      lio.start();
      lio_started = true;
    }

    while (lio.containsEnoughDataForSyncPackages()) {
      std::this_thread::sleep_for(std::chrono::milliseconds(50));
    }
    std::this_thread::sleep_for(std::chrono::milliseconds(500));
    lio.stop();
    writer.Finalize();
    spdlog::info("Finished realtime mapping: {} frames, {} output poses, {}",
                 frame_id, writer.result_count(), output_dir.string());
  } catch (const std::exception& error) {
    spdlog::error("Realtime mapping failed: {}", error.what());
    return 1;
  }
  return 0;
}
