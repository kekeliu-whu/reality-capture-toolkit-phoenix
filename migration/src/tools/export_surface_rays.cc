#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include "migration/proto_io.h"

namespace fs = std::filesystem;

namespace {

struct TileKey {
  std::int32_t x;
  std::int32_t y;
  std::int32_t z;

  bool operator==(const TileKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct TileKeyHash {
  std::size_t operator()(const TileKey& key) const noexcept {
    std::size_t seed = std::hash<std::int32_t>{}(key.x);
    seed ^= std::hash<std::int32_t>{}(key.y) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    seed ^= std::hash<std::int32_t>{}(key.z) + 0x9e3779b9U + (seed << 6U) +
            (seed >> 2U);
    return seed;
  }
};

// Version-2 ray-history layout consumed by navvis_recon_shard_surface_filter.
// One record is one exact (sensor origin, return endpoint) observation.
struct DiskRecordV2 {
  std::int32_t key_x;
  std::int32_t key_y;
  std::int32_t key_z;
  float xyz_x;
  float xyz_y;
  float xyz_z;
  float origin_x;
  float origin_y;
  float origin_z;
  float normal_x;
  float normal_y;
  float normal_z;
  float intensity;
  std::uint32_t count;
};
static_assert(sizeof(DiskRecordV2) == 56U, "Unexpected ray-history layout");

struct Options {
  fs::path lidar;
  fs::path trajectory;
  fs::path output_directory;
  float resolution = 0.01F;
  float minimum_range = 0.4F;
  float maximum_range = 30.0F;
  std::size_t maximum_buffer_records = 2'000'000U;
};

int FloorDivision(int value, int denominator) {
  int quotient = value / denominator;
  const int remainder = value % denominator;
  if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
    --quotient;
  }
  return quotient;
}

std::uint64_t TilecloudMortonKey(float x, float y, float z) {
  // Match the surface pipeline's 5 m Tilecloud ordering so its global
  // CompactOctree lattice is phase-aligned to the same first occupied tile.
  constexpr float kTileSize = 5.0F;
  constexpr std::uint32_t kCoordinateMask = (1U << 20U) - 1U;
  const std::array<std::int32_t, 3> coordinate{
      static_cast<std::int32_t>(std::floor(x / kTileSize)),
      static_cast<std::int32_t>(std::floor(y / kTileSize)),
      static_cast<std::int32_t>(std::floor(z / kTileSize))};
  std::uint64_t key = 0U;
  for (std::uint32_t bit = 0U; bit < 20U; ++bit) {
    for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
      const std::uint32_t value =
          static_cast<std::uint32_t>(coordinate[axis]) & kCoordinateMask;
      key |= static_cast<std::uint64_t>((value >> bit) & 1U)
             << (3U * bit + axis);
    }
  }
  return key;
}

Options ParseArguments(int argc, char** argv) {
  Options options;
  for (int index = 1; index < argc; ++index) {
    const std::string argument = argv[index];
    const auto value = [&]() -> std::string {
      if (++index >= argc) {
        throw std::invalid_argument("Missing value after " + argument);
      }
      return argv[index];
    };
    if (argument == "--lidar") {
      options.lidar = value();
    } else if (argument == "--trajectory") {
      options.trajectory = value();
    } else if (argument == "--output-directory") {
      options.output_directory = value();
    } else if (argument == "--resolution") {
      options.resolution = std::stof(value());
    } else if (argument == "--min-range") {
      options.minimum_range = std::stof(value());
    } else if (argument == "--max-range") {
      options.maximum_range = std::stof(value());
    } else if (argument == "--max-buffer-records") {
      options.maximum_buffer_records = std::stoull(value());
    } else if (argument == "--help") {
      std::cout
          << "Usage: export_surface_rays --lidar lidar_undist.dat "
             "--trajectory traj.dat --output-directory raw_shards "
             "[--resolution 0.01] [--min-range 0.4] [--max-range 30] "
             "[--max-buffer-records 2000000]\n";
      std::exit(0);
    } else {
      throw std::invalid_argument("Unknown argument: " + argument);
    }
  }
  if (options.lidar.empty() || options.trajectory.empty() ||
      options.output_directory.empty()) {
    throw std::invalid_argument(
        "--lidar, --trajectory and --output-directory are required");
  }
  if (!(options.resolution > 0.0F) || options.minimum_range < 0.0F ||
      !(options.maximum_range > options.minimum_range) ||
      options.maximum_buffer_records == 0U) {
    throw std::invalid_argument("Invalid resolution, range or buffer limit");
  }
  return options;
}

class ShardWriter {
 public:
  ShardWriter(fs::path output_directory, float resolution,
              std::size_t maximum_buffer_records)
      : output_directory_(std::move(output_directory)),
        resolution_(resolution),
        tile_voxels_(std::max(
            1, static_cast<int>(std::llround(10.0 / resolution_)))),
        maximum_buffer_records_(maximum_buffer_records) {
    fs::create_directories(output_directory_);
  }

  void Add(float x, float y, float z, float origin_x, float origin_y,
           float origin_z, float intensity) {
    const std::int32_t key_x =
        static_cast<std::int32_t>(std::floor(x / resolution_));
    const std::int32_t key_y =
        static_cast<std::int32_t>(std::floor(y / resolution_));
    const std::int32_t key_z =
        static_cast<std::int32_t>(std::floor(z / resolution_));
    const TileKey tile{FloorDivision(key_x, tile_voxels_),
                       FloorDivision(key_y, tile_voxels_),
                       FloorDivision(key_z, tile_voxels_)};
    buffers_[tile].push_back({key_x, key_y, key_z, x, y, z, origin_x,
                              origin_y, origin_z, 0.0F, 0.0F, 0.0F,
                              intensity, 1U});
    ++buffered_records_;
    ++total_records_;

    const std::uint64_t morton = TilecloudMortonKey(x, y, z);
    if (!has_anchor_ || morton < anchor_morton_) {
      anchor_morton_ = morton;
      anchor_ = {x, y, z};
      has_anchor_ = true;
    }
    if (buffered_records_ >= maximum_buffer_records_) {
      Flush();
    }
  }

  void Finish() {
    Flush();
    if (!has_anchor_) {
      throw std::runtime_error("No valid rays were written");
    }
    std::ofstream anchor(output_directory_ / "freespace_anchor.txt",
                         std::ios::trunc);
    if (!anchor) {
      throw std::runtime_error("Cannot write freespace_anchor.txt");
    }
    anchor << std::setprecision(9) << anchor_[0] << ' ' << anchor_[1] << ' '
           << anchor_[2] << '\n';
  }

  std::uint64_t total_records() const { return total_records_; }
  std::size_t shard_count() const { return shard_paths_.size(); }

 private:
  void Flush() {
    for (auto& item : buffers_) {
      const TileKey& tile = item.first;
      const fs::path path =
          output_directory_ /
          ("tile_" + std::to_string(tile.x) + "_" +
           std::to_string(tile.y) + "_" + std::to_string(tile.z) +
           ".raytile");
      std::ofstream output(path, std::ios::binary | std::ios::app);
      if (!output) {
        throw std::runtime_error("Cannot append ray shard: " + path.string());
      }
      const auto& records = item.second;
      output.write(reinterpret_cast<const char*>(records.data()),
                   static_cast<std::streamsize>(records.size() *
                                                sizeof(DiskRecordV2)));
      if (!output) {
        throw std::runtime_error("Failed while writing ray shard: " +
                                 path.string());
      }
      shard_paths_[tile] = true;
    }
    buffers_.clear();
    buffered_records_ = 0U;
  }

  fs::path output_directory_;
  float resolution_;
  int tile_voxels_;
  std::size_t maximum_buffer_records_;
  std::size_t buffered_records_ = 0U;
  std::uint64_t total_records_ = 0U;
  std::unordered_map<TileKey, std::vector<DiskRecordV2>, TileKeyHash> buffers_;
  std::unordered_map<TileKey, bool, TileKeyHash> shard_paths_;
  std::array<float, 3> anchor_{};
  std::uint64_t anchor_morton_ = std::numeric_limits<std::uint64_t>::max();
  bool has_anchor_ = false;
};

int Run(const Options& options) {
  if (!fs::is_regular_file(options.lidar)) {
    throw std::runtime_error("LiDAR input does not exist: " +
                             options.lidar.string());
  }
  if (!fs::is_regular_file(options.trajectory)) {
    throw std::runtime_error("Trajectory input does not exist: " +
                             options.trajectory.string());
  }
  if (fs::exists(options.output_directory) &&
      !fs::is_empty(options.output_directory)) {
    throw std::runtime_error("Output directory must be absent or empty: " +
                             options.output_directory.string());
  }

  const auto started = std::chrono::steady_clock::now();
  proto::PoseMsgList poses;
  if (!ReadPoseFile(options.trajectory.string(), poses) ||
      poses.pose_msgs().empty()) {
    throw std::runtime_error("Failed to read trajectory: " +
                             options.trajectory.string());
  }

  SequentialLidarFileReader<proto::LidarMsg> reader;
  if (!reader.Open(options.lidar.string())) {
    throw std::runtime_error("Failed to read LiDAR: " + options.lidar.string());
  }

  ShardWriter writer(options.output_directory, options.resolution,
                     options.maximum_buffer_records);
  std::uint64_t input_points = 0U;
  std::uint64_t rejected_nonfinite = 0U;
  std::uint64_t rejected_range = 0U;
  std::uint32_t maximum_intensity = 0U;
  std::size_t frame_index = 0U;
  const double minimum_range_squared =
      static_cast<double>(options.minimum_range) * options.minimum_range;
  const double maximum_range_squared =
      static_cast<double>(options.maximum_range) * options.maximum_range;

  std::shared_ptr<proto::LidarMsg> scan;
  while (reader.ReadNext(scan)) {
    if (frame_index >= static_cast<std::size_t>(poses.pose_msgs_size())) {
      throw std::runtime_error("More LiDAR frames than trajectory poses");
    }
    const auto& pose = poses.pose_msgs(static_cast<int>(frame_index));
    double qx = pose.rx();
    double qy = pose.ry();
    double qz = pose.rz();
    double qw = pose.rw();
    const double quaternion_norm =
        std::sqrt(qx * qx + qy * qy + qz * qz + qw * qw);
    if (!(quaternion_norm > 0.0) || !std::isfinite(quaternion_norm)) {
      throw std::runtime_error("Invalid quaternion at frame " +
                               std::to_string(frame_index));
    }
    qx /= quaternion_norm;
    qy /= quaternion_norm;
    qz /= quaternion_norm;
    qw /= quaternion_norm;
    const float origin_x = static_cast<float>(pose.tx());
    const float origin_y = static_cast<float>(pose.ty());
    const float origin_z = static_cast<float>(pose.tz());

    for (const auto& point : scan->points()) {
      ++input_points;
      const double x = point.x();
      const double y = point.y();
      const double z = point.z();
      if (!std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
        ++rejected_nonfinite;
        continue;
      }
      const double range_squared = x * x + y * y + z * z;
      if (range_squared < minimum_range_squared ||
          range_squared > maximum_range_squared) {
        ++rejected_range;
        continue;
      }

      // Rotate with q * point * conjugate(q), then translate into world space.
      const double tx = 2.0 * (qy * z - qz * y);
      const double ty = 2.0 * (qz * x - qx * z);
      const double tz = 2.0 * (qx * y - qy * x);
      const float world_x = static_cast<float>(
          x + qw * tx + (qy * tz - qz * ty) + pose.tx());
      const float world_y = static_cast<float>(
          y + qw * ty + (qz * tx - qx * tz) + pose.ty());
      const float world_z = static_cast<float>(
          z + qw * tz + (qx * ty - qy * tx) + pose.tz());
      if (!std::isfinite(world_x) || !std::isfinite(world_y) ||
          !std::isfinite(world_z)) {
        ++rejected_nonfinite;
        continue;
      }
      maximum_intensity = (std::max)(maximum_intensity, point.intensity());
      const float normalized_intensity =
          std::clamp(static_cast<float>(point.intensity()) / 255.0F, 0.0F,
                     1.0F);
      writer.Add(world_x, world_y, world_z, origin_x, origin_y, origin_z,
                 normalized_intensity);
    }

    ++frame_index;
    if (frame_index % 250U == 0U) {
      std::cerr << "Converted " << frame_index << '/'
                << poses.pose_msgs_size() << " frames (" << std::fixed
                << std::setprecision(1) << reader.getProgress() << "%), rays="
                << writer.total_records() << "\r";
    }
  }
  reader.Close();
  if (frame_index != static_cast<std::size_t>(poses.pose_msgs_size())) {
    throw std::runtime_error("Frame count mismatch: poses=" +
                             std::to_string(poses.pose_msgs_size()) +
                             ", clouds=" + std::to_string(frame_index));
  }
  writer.Finish();

  const double elapsed = std::chrono::duration<double>(
                             std::chrono::steady_clock::now() - started)
                             .count();
  std::ofstream stats(options.output_directory / "ray_export_stats.txt",
                      std::ios::trunc);
  stats << std::setprecision(12)
        << "frames=" << frame_index << '\n'
        << "input_points=" << input_points << '\n'
        << "output_rays=" << writer.total_records() << '\n'
        << "rejected_range=" << rejected_range << '\n'
        << "rejected_nonfinite=" << rejected_nonfinite << '\n'
        << "shards=" << writer.shard_count() << '\n'
        << "maximum_input_intensity=" << maximum_intensity << '\n'
        << "resolution_m=" << options.resolution << '\n'
        << "minimum_range_m=" << options.minimum_range << '\n'
        << "maximum_range_m=" << options.maximum_range << '\n'
        << "elapsed_seconds=" << elapsed << '\n';

  std::cerr << "\nRay export complete: frames=" << frame_index
            << ", input_points=" << input_points
            << ", output_rays=" << writer.total_records()
            << ", range_rejected=" << rejected_range
            << ", nonfinite_rejected=" << rejected_nonfinite
            << ", shards=" << writer.shard_count()
            << ", max_intensity=" << maximum_intensity
            << ", elapsed=" << std::fixed << std::setprecision(3) << elapsed
            << " s\n";
  return 0;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    return Run(ParseArguments(argc, argv));
  } catch (const std::exception& error) {
    std::cerr << "export_surface_rays: " << error.what() << '\n';
    return 1;
  }
}
