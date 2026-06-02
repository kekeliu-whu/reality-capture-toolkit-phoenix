/// @file offline_main.cpp
/// @brief Offline FAST-LIO2 runner for Windows — reads a .bag file and processes without ROS runtime.
///
/// Usage: fastlio_offline --bag <file.bag> --config <mid360.yaml> [--output <dir>]

#include <gflags/gflags.h>
#include <yaml-cpp/yaml.h>
#include <spdlog/spdlog.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <livox_ros_driver/CustomMsg.h>
#include <livox_ros_driver2/CustomMsg.h>

#include <iostream>
#include <fstream>
#include <vector>
#include <algorithm>
#include <set>
#include <chrono>
#include <filesystem>

#include <Eigen/Core>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/io/pcd_io.h>

#include "fastlio_config.h"

// Forward declarations of functions in laserMapping.cpp
// These are the entry points into the FAST-LIO2 pipeline
namespace fastlio_pipeline {

// Initialize the pipeline with config
void init(const fastlio::FastLioConfig& cfg);

// Feed a preprocessed LiDAR point cloud (from Livox CustomMsg)
void feed_lidar(const livox_ros_driver::CustomMsg::ConstPtr& msg);

// Feed an IMU message
void feed_imu(const sensor_msgs::Imu::ConstPtr& msg);

// Try to sync and process one LiDAR scan. Returns true if a scan was processed.
bool process_one_scan();

// Save final map and trajectory
void save_results(const std::string& output_dir);

// Check if shutdown was requested
bool is_done();

}  // namespace fastlio_pipeline

DEFINE_string(bag, "", "Path to input .bag file");
DEFINE_string(config, "", "Path to YAML configuration file (e.g., mid360.yaml)");
DEFINE_string(output, "./fastlio_output", "Output directory for trajectory and map");
DEFINE_bool(dump_topics, false, "List all topics in the bag file and exit");

using namespace std;

// ============================================================================
// Bag topic auto-detection
// ============================================================================
struct BagInfo {
  string lidar_topic;
  string imu_topic;
  bool lidar_is_livox = false;
};

static BagInfo DetectTopics(const string& bag_path) {
  BagInfo info;
  rosbag::Bag bag;
  bag.open(bag_path, rosbag::bagmode::Read);

  rosbag::View view(bag);

  for (const auto& conn : view.getConnections()) {
    string t = conn->topic;
    string dt = conn->datatype;

    if (dt == "livox_ros_driver/CustomMsg" || dt == "livox_ros_driver2/CustomMsg") {
      info.lidar_topic = t;
      info.lidar_is_livox = true;
      spdlog::info("Detected Livox LiDAR topic: {} [{}]", t, dt);
    } else if (dt == "sensor_msgs/PointCloud2" && info.lidar_topic.empty()) {
      if (t.find("lidar") != string::npos || t.find("hesai") != string::npos ||
          t.find("velodyne") != string::npos || t.find("ouster") != string::npos ||
          t.find("pandar") != string::npos) {
        info.lidar_topic = t;
        spdlog::info("Detected standard LiDAR topic: {}", t);
      }
    }

    if (dt == "sensor_msgs/Imu" && info.imu_topic.empty()) {
      if (t.find("imu") != string::npos) {
        info.imu_topic = t;
        spdlog::info("Detected IMU topic: {}", t);
      }
    }
  }

  // Fallback: use first IMU topic found
  if (info.imu_topic.empty()) {
    rosbag::View view2(bag);
    for (const auto& conn : view2.getConnections()) {
      string t = conn->topic;
      if (t.find("imu") != string::npos && conn->datatype == "sensor_msgs/Imu") {
        info.imu_topic = t;
        spdlog::info("Fallback IMU topic: {}", t);
        break;
      }
    }
  }

  bag.close();
  return info;
}

// ============================================================================
// Message types for pre-loaded bag data
// ============================================================================
enum class MsgType { IMU, LIDAR_LIVOX, OTHER };

struct BagMessage {
  double timestamp;
  MsgType type;
  boost::shared_ptr<sensor_msgs::Imu> imu_msg;
  boost::shared_ptr<livox_ros_driver::CustomMsg> livox_msg;
};

// ============================================================================
// Main
// ============================================================================
int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  // --- Dump topics mode ---
  if (FLAGS_dump_topics) {
    if (FLAGS_bag.empty()) {
      cerr << "Error: --bag is required with --dump_topics\n";
      return 1;
    }
    rosbag::Bag bag;
    bag.open(FLAGS_bag, rosbag::bagmode::Read);
    rosbag::View view(bag);
    cout << "\nTopics in bag:\n";
    for (const auto& conn : view.getConnections()) {
      cout << "  " << conn->topic << " [" << conn->datatype << "]\n";
    }
    bag.close();
    return 0;
  }

  // --- Validate inputs ---
  if (FLAGS_bag.empty()) {
    cerr << "Error: --bag is required\n";
    cerr << "Usage: fastlio_offline --bag <file.bag> --config <config.yaml> [--output <dir>]\n";
    return 1;
  }

  // --- Load config ---
  string config_path = FLAGS_config;
  if (config_path.empty()) {
    // Default: look for mid360.yaml next to the bag file
    filesystem::path bag_path(FLAGS_bag);
    auto parent = bag_path.parent_path();
    config_path = (parent / "mid360.yaml").string();
  }

  fastlio::FastLioConfig cfg = fastlio::FastLioConfig::FromYamlFile(config_path);
  cfg.output_dir = FLAGS_output;
  spdlog::info("Loaded config: {}", config_path);
  spdlog::info("LiDAR type: {}, scan_line: {}", cfg.lidar_type, cfg.scan_line);

  // --- Detect topics ---
  auto bag_info = DetectTopics(FLAGS_bag);
  if (bag_info.lidar_topic.empty()) {
    cerr << "Error: No LiDAR topic found in bag file\n";
    return 1;
  }
  if (bag_info.imu_topic.empty()) {
    cerr << "Error: No IMU topic found in bag file\n";
    return 1;
  }

  cfg.lid_topic = bag_info.lidar_topic;
  cfg.imu_topic = bag_info.imu_topic;

  // --- Read all messages from bag ---
  spdlog::info("Reading bag file: {} ...", FLAGS_bag);
  vector<BagMessage> messages;
  {
    rosbag::Bag bag;
    bag.open(FLAGS_bag, rosbag::bagmode::Read);
    rosbag::View view(bag);

    int imu_count = 0, lidar_count = 0;
    for (const auto& m : view) {
      BagMessage bm;

      if (m.getTopic() == bag_info.imu_topic) {
        auto msg = m.instantiate<sensor_msgs::Imu>();
        if (msg) {
          bm.timestamp = msg->header.stamp.toSec();
          bm.type = MsgType::IMU;
          bm.imu_msg = msg;
          imu_count++;
        }
      } else if (bag_info.lidar_is_livox && m.getTopic() == bag_info.lidar_topic) {
        // Try livox_ros_driver first, then livox_ros_driver2
        auto msg = m.instantiate<livox_ros_driver::CustomMsg>();
        if (msg) {
          bm.timestamp = msg->header.stamp.toSec();
          bm.type = MsgType::LIDAR_LIVOX;
          bm.livox_msg = msg;
          lidar_count++;
        } else {
          auto msg2 = m.instantiate<livox_ros_driver2::CustomMsg>();
          if (msg2) {
            // Convert livox_ros_driver2 -> livox_ros_driver (same data layout)
            auto converted = boost::make_shared<livox_ros_driver::CustomMsg>();
            converted->header = msg2->header;
            converted->timebase = msg2->timebase;
            converted->point_num = msg2->point_num;
            converted->lidar_id = msg2->lidar_id;
            converted->rsvd = msg2->rsvd;
            converted->points.resize(msg2->points.size());
            for (size_t p = 0; p < msg2->points.size(); p++) {
              converted->points[p].offset_time = msg2->points[p].offset_time;
              converted->points[p].x = msg2->points[p].x;
              converted->points[p].y = msg2->points[p].y;
              converted->points[p].z = msg2->points[p].z;
              converted->points[p].reflectivity = msg2->points[p].reflectivity;
              converted->points[p].tag = msg2->points[p].tag;
              converted->points[p].line = msg2->points[p].line;
            }
            bm.timestamp = msg2->header.stamp.toSec();
            bm.type = MsgType::LIDAR_LIVOX;
            bm.livox_msg = converted;
            lidar_count++;
          }
        }
      } else {
        continue;
      }

      messages.push_back(bm);
    }
    bag.close();

    spdlog::info("Loaded {} IMU messages, {} LiDAR scans", imu_count, lidar_count);
  }

  if (messages.empty()) {
    cerr << "Error: No messages found in bag\n";
    return 1;
  }

  // Sort by timestamp
  sort(messages.begin(), messages.end(),
       [](const BagMessage& a, const BagMessage& b) { return a.timestamp < b.timestamp; });

  // --- Setup output directory ---
  string output_dir = FLAGS_output;
  if (output_dir.back() != '/' && output_dir.back() != '\\') output_dir += "/";
  cfg.output_dir = output_dir;

  error_code ec;
  filesystem::create_directories(output_dir, ec);

  // --- Initialize FAST-LIO2 ---
  spdlog::info("Initializing FAST-LIO2 pipeline...");
  fastlio_pipeline::init(cfg);

  // --- Process messages ---
  spdlog::info("Starting processing ({} messages)...", messages.size());
  auto start_time = chrono::high_resolution_clock::now();

  size_t lidar_processed = 0;
  for (size_t i = 0; i < messages.size(); i++) {
    const auto& bm = messages[i];

    if (bm.type == MsgType::IMU) {
      fastlio_pipeline::feed_imu(bm.imu_msg);
    } else if (bm.type == MsgType::LIDAR_LIVOX) {
      fastlio_pipeline::feed_lidar(bm.livox_msg);

      // After feeding a LiDAR scan, try to process
      while (fastlio_pipeline::process_one_scan()) {
        lidar_processed++;
        if (lidar_processed % 100 == 0) {
          spdlog::info("Processed {} scans...", lidar_processed);
        }
      }
    }

    // Progress indicator
    if (i % 10000 == 0) {
      auto now = chrono::high_resolution_clock::now();
      auto elapsed = chrono::duration_cast<chrono::seconds>(now - start_time).count();
      if (elapsed > 0) {
        spdlog::info("Read {}/{} messages, {} scans processed ({:.1f}% in {}s)",
                     i, messages.size(), lidar_processed,
                     100.0 * i / messages.size(), elapsed);
      }
    }
  }

  // Process any remaining scans
  while (fastlio_pipeline::process_one_scan()) {
    lidar_processed++;
  }

  // --- Save results ---
  spdlog::info("Saving results to {} ...", output_dir);
  fastlio_pipeline::save_results(output_dir);

  auto end_time = chrono::high_resolution_clock::now();
  auto elapsed = chrono::duration_cast<chrono::seconds>(end_time - start_time).count();
  spdlog::info("Processing complete: {} LiDAR scans in {} seconds", lidar_processed, elapsed);
  spdlog::info("Results saved to: {}", output_dir);

  return 0;
}
