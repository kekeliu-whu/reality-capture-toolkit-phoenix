#include "navvis_recon/slam_batch_collator.hpp"
#include "navvis_recon/slam_frontend.hpp"
#include "navvis_recon/slam_imu_file.hpp"
#ifndef _WIN32
#include "navvis_recon/slam_rosbag.hpp"
#endif

#include <algorithm>
#include <array>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>

namespace {

struct Options {
  std::filesystem::path archive;
  std::filesystem::path imu_bag;
  std::filesystem::path imu_file;
  std::filesystem::path output;
  std::filesystem::path state_output;
  std::size_t batch_limit = 0U;
  std::size_t progress_every = 0U;
  navvis_recon::slam::FrontendConfig frontend;
};

std::size_t parseSize(const std::string& value, const char* option) {
  std::size_t consumed = 0U;
  const unsigned long long parsed = std::stoull(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument(std::string("invalid ") + option);
  }
  return static_cast<std::size_t>(parsed);
}

double parseDouble(const std::string& value, const char* option) {
  std::size_t consumed = 0U;
  const double parsed = std::stod(value, &consumed);
  if (consumed != value.size()) {
    throw std::invalid_argument(std::string("invalid ") + option);
  }
  return parsed;
}

navvis_recon::slam::Pose parsePose(const std::string& value) {
  std::array<double, 7> fields{};
  std::istringstream input(value);
  std::string token;
  std::size_t index = 0U;
  while (std::getline(input, token, ',')) {
    if (index >= fields.size()) {
      throw std::invalid_argument(
          "--initial-pose requires tx,ty,tz,qx,qy,qz,qw");
    }
    fields[index++] = parseDouble(token, "--initial-pose");
  }
  if (index != fields.size()) {
    throw std::invalid_argument(
        "--initial-pose requires tx,ty,tz,qx,qy,qz,qw");
  }
  return navvis_recon::slam::Pose{
      Eigen::Vector3d(fields[0], fields[1], fields[2]),
      Eigen::Quaterniond(fields[6], fields[3], fields[4], fields[5])};
}

Options parseOptions(const int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument(argument + " requires a value");
      }
      return argv[index];
    };
    if (argument == "--archive") {
      options.archive = value();
    } else if (argument == "--imu-bag") {
      options.imu_bag = value();
    } else if (argument == "--imu-file") {
      options.imu_file = value();
    } else if (argument == "--output") {
      options.output = value();
    } else if (argument == "--state-output") {
      options.state_output = value();
    } else if (argument == "--batch-limit") {
      options.batch_limit = parseSize(value(), "--batch-limit");
    } else if (argument == "--progress-every") {
      options.progress_every = parseSize(value(), "--progress-every");
    } else if (argument == "--threads") {
      options.frontend.icp_threads =
          static_cast<int>(parseSize(value(), "--threads"));
    } else if (argument == "--initial-pose") {
      options.frontend.initial_tracking_pose = parsePose(value());
    } else if (argument == "--motion-max-time") {
      options.frontend.motion_filter_maximum_time_s =
          parseDouble(value(), "--motion-max-time");
    } else if (argument == "--motion-max-distance") {
      options.frontend.motion_filter_maximum_distance_m =
          parseDouble(value(), "--motion-max-distance");
    } else if (argument == "--motion-max-angle-deg") {
      options.frontend.motion_filter_maximum_angle_rad =
          parseDouble(value(), "--motion-max-angle-deg") *
          3.14159265358979323846 / 180.0;
    } else if (argument == "--retain-all") {
      options.frontend.motion_filter_maximum_time_s = 0.0;
      options.frontend.motion_filter_maximum_distance_m = 0.0;
      options.frontend.motion_filter_maximum_angle_rad = 0.0;
    } else if (argument == "--help") {
      std::cout
          << "usage: navvis_recon_slam --archive NVSLAM6 "
             "(--imu-bag BAG | --imu-file RAW_IMU) "
             "--output TRAJECTORY.csv [--state-output FRONTEND.bin] "
             "[--batch-limit N] [--threads N] "
             "[--initial-pose tx,ty,tz,qx,qy,qz,qw] "
             "[--progress-every N] [--retain-all] "
             "[--motion-max-time S] [--motion-max-distance M] "
             "[--motion-max-angle-deg DEG]\n";
      std::exit(EXIT_SUCCESS);
    } else {
      throw std::invalid_argument("unknown option: " + argument);
    }
  }
  if (options.archive.empty() || options.output.empty() ||
      (options.imu_bag.empty() == options.imu_file.empty())) {
    throw std::invalid_argument(
        "--archive, --output and exactly one of --imu-bag/--imu-file are required");
  }
#ifdef _WIN32
  if (!options.imu_bag.empty()) {
    throw std::invalid_argument(
        "--imu-bag requires ROS1; export it and pass --imu-file on Windows");
  }
#endif
  return options;
}

double secondsSince(const std::chrono::steady_clock::time_point start) {
  return std::chrono::duration<double>(std::chrono::steady_clock::now() - start)
      .count();
}

void writeTrajectory(const std::filesystem::path& path,
                     const navvis_recon::slam::FrontendResult& result) {
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write SLAM trajectory " + path.string());
  }
  output << "index,timestamp_ns,tx,ty,tz,qx,qy,qz,qw,icp_correspondences,"
            "icp_iterations,icp_fitness_m\n";
  output << std::setprecision(17);
  for (const navvis_recon::slam::FrontendNode& node : result.nodes) {
    output << node.index << ',' << node.timestamp_ns << ','
           << node.local_pose.translation.x() << ','
           << node.local_pose.translation.y() << ','
           << node.local_pose.translation.z() << ','
           << node.local_pose.rotation.x() << ','
           << node.local_pose.rotation.y() << ','
           << node.local_pose.rotation.z() << ','
           << node.local_pose.rotation.w() << ','
           << node.scan_match.correspondence_count << ','
           << node.scan_match.iteration_count << ','
           << node.scan_match.plane_fitness_m << '\n';
  }
  if (!output) {
    throw std::runtime_error("failed while writing SLAM trajectory " +
                             path.string());
  }
}

template <typename Value>
void writeScalar(std::ofstream& output, const Value& value) {
  static_assert(std::is_trivially_copyable_v<Value>);
  output.write(reinterpret_cast<const char*>(&value), sizeof(value));
}

void writeBytes(std::ofstream& output, const void* data,
                const std::size_t size) {
  if (size != 0U) {
    output.write(static_cast<const char*>(data),
                 static_cast<std::streamsize>(size));
  }
}

void writePose(std::ofstream& output, const navvis_recon::slam::Pose& pose) {
  const std::array<double, 7> values{
      pose.translation.x(), pose.translation.y(), pose.translation.z(),
      pose.rotation.x(), pose.rotation.y(), pose.rotation.z(),
      pose.rotation.w()};
  writeBytes(output, values.data(), values.size() * sizeof(double));
}

void writeVector3f(std::ofstream& output,
                   const std::vector<Eigen::Vector3f>& values) {
  static_assert(sizeof(Eigen::Vector3f) == 3U * sizeof(float));
  writeBytes(output, values.data(), values.size() * sizeof(Eigen::Vector3f));
}

// Versioned, little-endian production handoff. It intentionally contains no
// official/reference fields: every byte is generated by the current frontend.
void writeFrontendState(const std::filesystem::path& path,
                        const navvis_recon::slam::FrontendResult& result) {
  const std::uint16_t endian_probe = 1U;
  if (*reinterpret_cast<const std::uint8_t*>(&endian_probe) != 1U) {
    throw std::runtime_error("SLAM frontend state requires little-endian host");
  }
  if (!path.parent_path().empty()) {
    std::filesystem::create_directories(path.parent_path());
  }
  std::ofstream output(path, std::ios::binary | std::ios::trunc);
  if (!output) {
    throw std::runtime_error("cannot write SLAM frontend state " + path.string());
  }
  constexpr std::array<char, 16> magic{
      'N', 'V', 'C', 'R', 'S', 'L', 'A', 'M',
      'S', 'T', 'A', 'T', 'E', '0', '1', '\0'};
  writeBytes(output, magic.data(), magic.size());
  writeScalar(output, std::uint32_t{1U});
  writeScalar(output, std::uint32_t{0x01020304U});
  writeScalar(output, static_cast<std::uint64_t>(result.nodes.size()));
  writeScalar(output, static_cast<std::uint64_t>(result.submaps.size()));

  for (const navvis_recon::slam::FrontendNode& node : result.nodes) {
    writeScalar(output, node.timestamp_ns);
    writePose(output, node.local_pose);
    const std::array<double, 3> gravity{
        node.gravity_observation.x(), node.gravity_observation.y(),
        node.gravity_observation.z()};
    writeBytes(output, gravity.data(), gravity.size() * sizeof(double));
    writeScalar(output, static_cast<std::uint64_t>(node.points.size()));
    writeVector3f(output, node.points);
  }

  for (const navvis_recon::slam::SubmapSummary& submap : result.submaps) {
    writeScalar(output, static_cast<std::uint64_t>(submap.index));
    writeScalar(output, submap.start_timestamp_ns);
    writeScalar(output, submap.end_timestamp_ns);
    writePose(output, submap.local_pose);
    writeScalar(output, static_cast<std::uint8_t>(submap.finished ? 1U : 0U));
    writeScalar(output,
                static_cast<std::uint64_t>(submap.node_indices.size()));
    for (const std::size_t index : submap.node_indices) {
      writeScalar(output, static_cast<std::uint64_t>(index));
    }
    for (std::size_t level = 0; level < 3U; ++level) {
      if (submap.surfel_points[level].size() !=
          submap.surfel_normals[level].size()) {
        throw std::runtime_error("SLAM submap surfel point/normal mismatch");
      }
      writeScalar(output, static_cast<std::uint64_t>(
                              submap.surfel_points[level].size()));
      writeVector3f(output, submap.surfel_points[level]);
      writeVector3f(output, submap.surfel_normals[level]);
    }
    if (submap.probability_grid_indices.size() !=
        submap.probability_grid_values.size()) {
      throw std::runtime_error("SLAM probability-grid payload mismatch");
    }
    writeScalar(output, static_cast<std::uint64_t>(
                            submap.probability_grid_indices.size()));
    writeBytes(output, submap.probability_grid_indices.data(),
               submap.probability_grid_indices.size() *
                   sizeof(submap.probability_grid_indices.front()));
    writeBytes(output, submap.probability_grid_values.data(),
               submap.probability_grid_values.size() * sizeof(std::uint16_t));
  }
  if (!output) {
    throw std::runtime_error("failed while writing SLAM frontend state " +
                             path.string());
  }
}

navvis_recon::slam::Pose predictorInitialPose(
    const navvis_recon::slam::Pose& first_node_pose,
    const std::int64_t first_all_sources_ns,
    const std::int64_t first_node_ns,
    const std::vector<navvis_recon::slam::ImuSample>& imu) {
  navvis_recon::slam::RawImuTracker tracker(imu);
  Eigen::Quaterniond tracker_at_start;
  Eigen::Quaterniond tracker_at_first_node;
  if (first_node_ns < first_all_sources_ns) {
    tracker_at_first_node = tracker.advance(first_node_ns).normalized();
    tracker_at_start = tracker.advance(first_all_sources_ns).normalized();
  } else {
    tracker_at_start = tracker.advance(first_all_sources_ns).normalized();
    tracker_at_first_node = tracker.advance(first_node_ns).normalized();
  }
  const Eigen::Quaterniond start_to_first_node =
      (tracker_at_start.conjugate() * tracker_at_first_node).normalized();
  const Eigen::Quaterniond rotation =
      (first_node_pose.rotation.normalized() *
       start_to_first_node.conjugate()).normalized();
  return navvis_recon::slam::Pose{first_node_pose.translation, rotation};
}

}  // namespace

int main(int argc, char** argv) {
  try {
    const Options options = parseOptions(argc, argv);
    const auto total_started = std::chrono::steady_clock::now();
    navvis_recon::slam::SlamBatchCollator collator(options.archive);
    std::size_t batch_count =
        options.batch_limit == 0U
            ? collator.batchCount()
            : std::min(options.batch_limit, collator.batchCount());
    const std::int64_t end_timestamp_ns =
        collator.batchTimestampsNs().at(batch_count - 1U);
    std::vector<navvis_recon::slam::ImuSample> imu;
    if (!options.imu_file.empty()) {
      imu = navvis_recon::slam::loadRawImuFile(
          options.imu_file, end_timestamp_ns);
    } else {
#ifndef _WIN32
      imu = navvis_recon::slam::loadRawImuRosbag(
          options.imu_bag, "/imu/imu_raw/data", end_timestamp_ns);
#endif
    }
    const auto supported_end = std::upper_bound(
        collator.batchTimestampsNs().begin(),
        collator.batchTimestampsNs().begin() + batch_count,
        imu.back().timestamp_ns);
    const std::size_t supported_batch_count = static_cast<std::size_t>(
        supported_end - collator.batchTimestampsNs().begin());
    if (supported_batch_count < batch_count) {
      std::cerr << "SLAM C++ trimming "
                << (batch_count - supported_batch_count)
                << " trailing batches beyond IMU support; last_imu_ns="
                << imu.back().timestamp_ns << '\n';
      batch_count = supported_batch_count;
    }
    if (batch_count == 0U) {
      throw std::runtime_error("no lidar batches fall within IMU support");
    }
    const double input_seconds = secondsSince(total_started);

    navvis_recon::slam::Pose predictor_initial =
        navvis_recon::slam::Pose::identity();
    if (options.frontend.initial_tracking_pose.has_value()) {
      predictor_initial = predictorInitialPose(
          *options.frontend.initial_tracking_pose,
          collator.firstAllSourcesTimestampNs(),
          collator.batchTimestampsNs().front(), imu);
    }
    navvis_recon::slam::ImuPosePredictor predictor(
        std::move(imu), options.frontend.maximum_range_m, predictor_initial);
    navvis_recon::slam::Frontend frontend(options.frontend);
    const auto frontend_started = std::chrono::steady_clock::now();
    for (std::size_t index = 0; index < batch_count; ++index) {
      frontend.process(collator.next(), predictor);
      if (options.progress_every != 0U &&
          (index + 1U) % options.progress_every == 0U) {
        std::cerr << "SLAM C++ progress: " << (index + 1U) << '/'
                  << batch_count << " batches; "
                  << secondsSince(frontend_started) << " s\n";
      }
    }
    navvis_recon::slam::FrontendResult result = frontend.finish();
    const double frontend_seconds = secondsSince(frontend_started);
    writeTrajectory(options.output, result);
    if (!options.state_output.empty()) {
      writeFrontendState(options.state_output, result);
    }
    std::cout << std::setprecision(9)
              << "SLAM C++ complete: batches=" << result.processed_batches
              << "; nodes=" << result.nodes.size()
              << "; motion_filtered=" << result.motion_filtered_batches
              << "; submaps=" << result.submaps.size()
              << "; input=" << input_seconds << " s"
              << "; frontend=" << frontend_seconds << " s"
              << "; total=" << secondsSince(total_started) << " s\n";
    for (const navvis_recon::slam::SubmapSummary& submap : result.submaps) {
      std::cout << "submap[" << submap.index << "]: nodes="
                << submap.node_indices.size() << "; surfels="
                << submap.surfel_counts[0] << ',' << submap.surfel_counts[1]
                << ',' << submap.surfel_counts[2] << "; grid="
                << submap.probability_grid_cells << '\n';
    }
    const auto& timing = result.timing;
    std::cout << "SLAM phase timing: deskew=" << timing.deskew_seconds
              << " s; centroid=" << timing.centroid_filter_seconds
              << " s; predict/icp=" << timing.prediction_and_icp_seconds
              << " s; correction/motion="
              << timing.correction_and_motion_filter_seconds
              << " s; node-filter=" << timing.node_filter_seconds
              << " s; submap-insert=" << timing.submap_insertion_seconds
              << " s; submap-transform=" << timing.submap_transform_seconds
              << " s; surfels=" << timing.surfel_update_seconds[0] << ','
              << timing.surfel_update_seconds[1] << ','
              << timing.surfel_update_seconds[2]
              << " s; probability-grid=" << timing.probability_grid_seconds
              << " s; finish=" << timing.finish_seconds << " s\n";
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << "navvis_recon_slam: " << error.what() << '\n';
    return EXIT_FAILURE;
  }
}
