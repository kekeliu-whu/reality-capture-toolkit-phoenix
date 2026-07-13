#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(input_dir, R"(D:\Users\rick\Desktop\tmp\hesai\data_20260707_173104\Raw Data)", "Kosmo/Hesai Raw Data directory");
DEFINE_string(output_dir, R"(D:\output-hs)", "Output dir to save converted data");
DEFINE_bool(skip_calibration, false, "Skip calib_info.yaml conversion");
DEFINE_bool(skip_imu, false, "Skip ext_imu_*.mcap conversion");
DEFINE_bool(skip_lidar, false, "Skip lidar_imu_*.mcap conversion");

namespace {

constexpr uint8_t kMcapOpHeader  = 0x01;
constexpr uint8_t kMcapOpChannel = 0x04;
constexpr uint8_t kMcapOpMessage = 0x05;
constexpr uint8_t kMcapOpDataEnd = 0x0f;

constexpr size_t kPointStride = 27;
constexpr size_t kImuStride   = 34;

struct McapChannel {
  uint16_t id = 0;
  std::string topic;
  std::string message_encoding;
};

struct McapMessage {
  uint16_t channel_id = 0;
  uint64_t log_time = 0;
  uint64_t publish_time = 0;
  std::vector<uint8_t> data;
};

struct PcWindow {
  uint64_t begin_ns = 0;
  uint64_t end_ns = 0;
};

template <typename T>
T ReadLe(const uint8_t* data) {
  static_assert(std::is_integral_v<T> || std::is_floating_point_v<T>);
  T value;
  std::memcpy(&value, data, sizeof(T));
  return value;
}

uint64_t FileSize(std::ifstream& in) {
  const auto current = in.tellg();
  in.seekg(0, std::ios::end);
  const auto size = in.tellg();
  in.seekg(current, std::ios::beg);
  return static_cast<uint64_t>(static_cast<std::streamoff>(size));
}

uint64_t CurrentPos(std::ifstream& in) {
  return static_cast<uint64_t>(static_cast<std::streamoff>(in.tellg()));
}

class RecordCursor {
 public:
  explicit RecordCursor(std::vector<uint8_t> data) : data_(std::move(data)) {}

  uint16_t U16() {
    Ensure(2);
    uint16_t value = ReadLe<uint16_t>(data_.data() + pos_);
    pos_ += 2;
    return value;
  }

  uint32_t U32() {
    Ensure(4);
    uint32_t value = ReadLe<uint32_t>(data_.data() + pos_);
    pos_ += 4;
    return value;
  }

  uint64_t U64() {
    Ensure(8);
    uint64_t value = ReadLe<uint64_t>(data_.data() + pos_);
    pos_ += 8;
    return value;
  }

  std::string String() {
    const uint32_t len = U32();
    Ensure(len);
    std::string value(reinterpret_cast<const char*>(data_.data() + pos_), len);
    pos_ += len;
    return value;
  }

  void Skip(size_t bytes) {
    Ensure(bytes);
    pos_ += bytes;
  }

  std::vector<uint8_t> RemainingBytes() {
    std::vector<uint8_t> bytes(data_.begin() + static_cast<std::ptrdiff_t>(pos_), data_.end());
    pos_ = data_.size();
    return bytes;
  }

 private:
  void Ensure(size_t bytes) const {
    if (bytes > data_.size() - pos_) {
      throw std::runtime_error("MCAP record is truncated");
    }
  }

  std::vector<uint8_t> data_;
  size_t pos_ = 0;
};

class McapReader {
 public:
  explicit McapReader(const std::filesystem::path& path) : path_(path) {}

  template <typename Callback>
  void ReadMessages(Callback&& callback) {
    std::ifstream in(path_, std::ios::binary);
    if (!in.is_open()) {
      throw std::runtime_error("Failed to open MCAP: " + path_.string());
    }

    const uint64_t size = FileSize(in);
    char magic[8] = {};
    in.read(magic, sizeof(magic));
    if (std::memcmp(magic, "\x89MCAP0\r\n", 8) != 0) {
      throw std::runtime_error("Invalid MCAP magic: " + path_.string());
    }

    while (CurrentPos(in) + 9 <= size) {
      uint8_t opcode = 0;
      uint64_t length = 0;
      in.read(reinterpret_cast<char*>(&opcode), 1);
      in.read(reinterpret_cast<char*>(&length), 8);
      if (!in || length > size || CurrentPos(in) + length > size) {
        throw std::runtime_error("Invalid MCAP record length in " + path_.string());
      }

      std::vector<uint8_t> payload(static_cast<size_t>(length));
      if (length > 0) {
        in.read(reinterpret_cast<char*>(payload.data()), static_cast<std::streamsize>(payload.size()));
      }

      if (opcode == kMcapOpHeader) {
        continue;
      } else if (opcode == kMcapOpChannel) {
        ParseChannel(std::move(payload));
      } else if (opcode == kMcapOpMessage) {
        McapMessage msg = ParseMessage(std::move(payload));
        const auto it = channels_.find(msg.channel_id);
        if (it != channels_.end()) {
          callback(it->second, msg);
        }
      } else if (opcode == kMcapOpDataEnd) {
        break;
      }
    }
  }

 private:
  void ParseChannel(std::vector<uint8_t> payload) {
    RecordCursor cursor(std::move(payload));
    McapChannel channel;
    channel.id = cursor.U16();
    cursor.U16();  // schema_id
    channel.topic = cursor.String();
    channel.message_encoding = cursor.String();

    const uint32_t metadata_count = cursor.U32();
    for (uint32_t i = 0; i < metadata_count; ++i) {
      cursor.String();
      cursor.String();
    }
    channels_[channel.id] = std::move(channel);
  }

  McapMessage ParseMessage(std::vector<uint8_t> payload) {
    RecordCursor cursor(std::move(payload));
    McapMessage msg;
    msg.channel_id = cursor.U16();
    cursor.U32();  // sequence
    msg.log_time = cursor.U64();
    msg.publish_time = cursor.U64();
    msg.data = cursor.RemainingBytes();
    return msg;
  }

  std::filesystem::path path_;
  std::map<uint16_t, McapChannel> channels_;
};

std::vector<std::filesystem::path> GlobFiles(const std::filesystem::path& dir, const std::string& prefix, const std::string& extension) {
  std::vector<std::filesystem::path> files;
  if (!std::filesystem::exists(dir)) {
    return files;
  }
  for (const auto& entry : std::filesystem::directory_iterator(dir)) {
    if (!entry.is_regular_file()) {
      continue;
    }
    const auto filename = entry.path().filename().string();
    if (entry.path().extension() == extension && filename.rfind(prefix, 0) == 0) {
      files.push_back(entry.path());
    }
  }
  std::sort(files.begin(), files.end());
  return files;
}

std::vector<PcWindow> ReadPcWindows(const std::filesystem::path& input_dir) {
  std::vector<PcWindow> windows;
  for (const auto& path : GlobFiles(input_dir, "pc_windows_", ".mcap")) {
    spdlog::info("Reading scan windows from {}", path.string());
    McapReader(path).ReadMessages([&](const McapChannel& channel, const McapMessage& msg) {
      if (channel.topic != "/lidar/pc_window") {
        return;
      }
      if (msg.data.size() != 16) {
        spdlog::warn("Ignoring malformed pc_window message with {} bytes", msg.data.size());
        return;
      }
      PcWindow window;
      window.begin_ns = ReadLe<uint64_t>(msg.data.data());
      window.end_ns   = ReadLe<uint64_t>(msg.data.data() + 8);
      if (window.begin_ns < window.end_ns) {
        windows.push_back(window);
      }
    });
  }
  std::sort(windows.begin(), windows.end(), [](const PcWindow& a, const PcWindow& b) { return a.begin_ns < b.begin_ns; });
  windows.erase(std::unique(windows.begin(), windows.end(), [](const PcWindow& a, const PcWindow& b) {
                  return a.begin_ns == b.begin_ns && a.end_ns == b.end_ns;
                }),
                windows.end());
  spdlog::info("Loaded {} lidar scan windows", windows.size());
  return windows;
}

bool IsValidPoint(float x, float y, float z, uint64_t timestamp_ns) {
  if (timestamp_ns == 0 || !std::isfinite(x) || !std::isfinite(y) || !std::isfinite(z)) {
    return false;
  }
  return x != 0.0f || y != 0.0f || z != 0.0f;
}

void FlushLidarScan(const std::shared_ptr<proto::LidarMsg>& lidar_msg, SequentialLidarFileWriter<proto::LidarMsg>* writer, size_t* scan_count,
                    size_t* point_count) {
  if (lidar_msg->points_size() == 0) {
    return;
  }
  *point_count += static_cast<size_t>(lidar_msg->points_size());
  writer->Write(lidar_msg);
  ++(*scan_count);
}

bool ConvertLidar(const std::filesystem::path& input_dir, const std::filesystem::path& output_dir) {
  const auto windows = ReadPcWindows(input_dir);
  if (windows.empty()) {
    spdlog::error("No pc_windows_*.mcap scan windows found in {}", input_dir.string());
    return false;
  }

  const auto lidar_files = GlobFiles(input_dir, "lidar_imu_", ".mcap");
  if (lidar_files.empty()) {
    spdlog::error("No lidar_imu_*.mcap files found in {}", input_dir.string());
    return false;
  }

  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  if (!lidar_writer.Open((output_dir / "lidar.dat").string())) {
    spdlog::error("Failed to open lidar.dat for writing");
    return false;
  }

  size_t window_index = 0;
  size_t scan_count = 0;
  size_t point_count = 0;
  size_t packet_count = 0;
  size_t skipped_points = 0;
  auto current_scan = std::make_shared<proto::LidarMsg>();

  auto advance_to = [&](uint64_t point_time_ns) {
    while (window_index < windows.size() && point_time_ns >= windows[window_index].end_ns) {
      FlushLidarScan(current_scan, &lidar_writer, &scan_count, &point_count);
      current_scan = std::make_shared<proto::LidarMsg>();
      ++window_index;
    }
  };

  for (const auto& path : lidar_files) {
    spdlog::info("Converting lidar MCAP {}", path.string());
    McapReader(path).ReadMessages([&](const McapChannel& channel, const McapMessage& msg) {
      if (channel.topic != "/lidar/pointcloud") {
        return;
      }
      ++packet_count;
      if (msg.data.size() % kPointStride != 0) {
        spdlog::warn("Ignoring malformed lidar packet with {} bytes", msg.data.size());
        return;
      }

      for (size_t offset = 0; offset + kPointStride <= msg.data.size(); offset += kPointStride) {
        const uint8_t* point_data = msg.data.data() + offset;
        const float x = ReadLe<float>(point_data);
        const float y = ReadLe<float>(point_data + 4);
        const float z = ReadLe<float>(point_data + 8);
        const float intensity = ReadLe<float>(point_data + 12);
        const uint64_t timestamp_ns = ReadLe<uint64_t>(point_data + 19);

        if (!IsValidPoint(x, y, z, timestamp_ns)) {
          ++skipped_points;
          continue;
        }

        advance_to(timestamp_ns);
        if (window_index >= windows.size()) {
          ++skipped_points;
          continue;
        }
        if (timestamp_ns < windows[window_index].begin_ns) {
          ++skipped_points;
          continue;
        }

        auto point = current_scan->add_points();
        point->set_timestamp(static_cast<double>(timestamp_ns) * 1e-9);
        point->set_x(x);
        point->set_y(y);
        point->set_z(z);
        point->set_intensity(static_cast<uint32_t>(std::max(0.0f, intensity)));
      }
    });
  }

  FlushLidarScan(current_scan, &lidar_writer, &scan_count, &point_count);
  spdlog::info("Wrote lidar.dat: {} scans, {} points, {} packets, {} skipped points", scan_count, point_count, packet_count, skipped_points);
  return scan_count > 0;
}

bool ConvertImu(const std::filesystem::path& input_dir, const std::filesystem::path& output_dir) {
  const auto imu_files = GlobFiles(input_dir, "ext_imu_", ".mcap");
  if (imu_files.empty()) {
    spdlog::error("No ext_imu_*.mcap files found in {}", input_dir.string());
    return false;
  }

  proto::ImuMsgList imu_msg_list;
  size_t skipped = 0;
  for (const auto& path : imu_files) {
    spdlog::info("Converting IMU MCAP {}", path.string());
    McapReader(path).ReadMessages([&](const McapChannel& channel, const McapMessage& msg) {
      if (channel.topic != "/lidar/imu/ext") {
        return;
      }
      if (msg.data.size() != kImuStride) {
        ++skipped;
        return;
      }

      const uint8_t* data = msg.data.data();
      const uint64_t timestamp_ns = ReadLe<uint64_t>(data);
      auto imu_msg = imu_msg_list.add_imu_msgs();
      imu_msg->set_timestamp(static_cast<double>(timestamp_ns) * 1e-9);
      imu_msg->set_gx(ReadLe<float>(data + 8));
      imu_msg->set_gy(ReadLe<float>(data + 12));
      imu_msg->set_gz(ReadLe<float>(data + 16));
      imu_msg->set_ax(ReadLe<float>(data + 20));
      imu_msg->set_ay(ReadLe<float>(data + 24));
      imu_msg->set_az(ReadLe<float>(data + 28));
    });
  }

  if (!WriteImuFile((output_dir / "imu.dat").string(), imu_msg_list)) {
    spdlog::error("Failed to write imu.dat");
    return false;
  }
  spdlog::info("Wrote imu.dat: {} messages, {} skipped", imu_msg_list.imu_msgs_size(), skipped);
  return true;
}

YAML::Node RequireNode(const YAML::Node& parent, const std::string& key) {
  const YAML::Node node = parent[key];
  if (!node) {
    throw std::runtime_error("Missing required YAML node: " + key);
  }
  return node;
}

Eigen::Matrix4d ReadMatrix4(const YAML::Node& node) {
  if (!node || !node.IsSequence() || node.size() != 4) {
    throw std::runtime_error("Expected 4x4 matrix");
  }
  Eigen::Matrix4d matrix;
  for (size_t r = 0; r < 4; ++r) {
    if (!node[r].IsSequence() || node[r].size() != 4) {
      throw std::runtime_error("Expected 4x4 matrix row");
    }
    for (size_t c = 0; c < 4; ++c) {
      matrix(static_cast<int>(r), static_cast<int>(c)) = node[r][c].as<double>();
    }
  }
  return matrix;
}

void SetExtrinsicFromMatrix(const Eigen::Matrix4d& transform, proto::SensorExtrinsic* extrinsic) {
  Eigen::Quaterniond q(transform.block<3, 3>(0, 0));
  q.normalize();
  extrinsic->set_rw(q.w());
  extrinsic->set_rx(q.x());
  extrinsic->set_ry(q.y());
  extrinsic->set_rz(q.z());
  extrinsic->set_tx(transform(0, 3));
  extrinsic->set_ty(transform(1, 3));
  extrinsic->set_tz(transform(2, 3));
}

bool ConvertCalibration(const std::filesystem::path& input_dir, const std::filesystem::path& output_dir) {
  const auto calib_path = input_dir / "calib_info.yaml";
  if (!std::filesystem::exists(calib_path)) {
    spdlog::error("Calibration file not found: {}", calib_path.string());
    return false;
  }

  try {
    YAML::Node root = YAML::LoadFile(calib_path.string());
    proto::SensorCalib calib;
    calib.set_has_encoder(false);

    const YAML::Node intrinsic = RequireNode(root, "intrinsic");
    const YAML::Node extrinsic = RequireNode(root, "extrinsic");

    SetExtrinsicFromMatrix(ReadMatrix4(RequireNode(RequireNode(extrinsic, "T_lidar_2_imu"), "T")), calib.mutable_lidar_to_encoder());

    for (const std::string cam_name : {"cam0", "cam1", "cam2"}) {
      const YAML::Node k_node = intrinsic[cam_name];
      const YAML::Node d_node = intrinsic[cam_name + "_distortion"];
      const YAML::Node t_node = extrinsic["T_lidar_2_" + cam_name];
      if (!k_node || !d_node || !t_node) {
        continue;
      }

      auto cam = calib.add_camera_param();
      cam->set_name(cam_name);
      cam->set_fx(k_node[0][0].as<double>());
      cam->set_fy(k_node[1][1].as<double>());
      cam->set_cx(k_node[0][2].as<double>());
      cam->set_cy(k_node[1][2].as<double>());
      cam->set_k1(d_node[0].as<double>());
      cam->set_k2(d_node[1].as<double>());
      cam->set_k3(d_node[2].as<double>());
      cam->set_k4(d_node[3].as<double>());
      SetExtrinsicFromMatrix(ReadMatrix4(RequireNode(t_node, "T")), cam->mutable_extrinsic());
    }

    if (!WriteSensorCalibFile((output_dir / "calibration.dat").string(), calib)) {
      spdlog::error("Failed to write calibration.dat");
      return false;
    }
    spdlog::info("Wrote calibration.dat with {} camera params", calib.camera_param_size());
    return true;
  } catch (const std::exception& e) {
    spdlog::error("Failed to convert calibration: {}", e.what());
    return false;
  }
}

}  // namespace

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  const std::filesystem::path input_dir = FLAGS_input_dir;
  const std::filesystem::path output_dir = FLAGS_output_dir;
  if (!std::filesystem::exists(input_dir)) {
    spdlog::error("Input directory does not exist: {}", input_dir.string());
    return 1;
  }
  std::filesystem::create_directories(output_dir);

  bool ok = true;
  if (!FLAGS_skip_calibration) {
    ok = ConvertCalibration(input_dir, output_dir) && ok;
  }
  if (!FLAGS_skip_imu) {
    ok = ConvertImu(input_dir, output_dir) && ok;
  }
  if (!FLAGS_skip_lidar) {
    ok = ConvertLidar(input_dir, output_dir) && ok;
  }

  spdlog::info("done.");
  return ok ? 0 : 1;
}
