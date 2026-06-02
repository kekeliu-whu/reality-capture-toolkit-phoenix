/// @file fastlio_config.h
/// @brief Configuration struct for FAST-LIO2 — replaces ROS parameter server.
///        Loaded from YAML (mid360.yaml) via yaml-cpp.

#ifndef FASTLIO_CONFIG_H
#define FASTLIO_CONFIG_H

#include <string>
#include <vector>
#include <yaml-cpp/yaml.h>

namespace fastlio {

struct FastLioConfig {
  // --- common ---
  std::string lid_topic = "/livox/lidar";
  std::string imu_topic = "/livox/imu";
  bool   time_sync_en = false;
  double time_offset_lidar_to_imu = 0.0;

  // --- preprocess ---
  int    lidar_type = 1;      // 1=AVIA, 2=VELO16, 3=OUST64, 4=MARSIM
  int    scan_line = 4;
  double blind = 0.5;
  int    point_filter_num = 2;
  bool   feature_extract_enable = false;
  int    timestamp_unit = 2;   // US
  int    scan_rate = 10;

  // --- mapping ---
  double acc_cov = 0.1;
  double gyr_cov = 0.1;
  double b_acc_cov = 0.0001;
  double b_gyr_cov = 0.0001;
  double fov_degree = 360.0;
  double det_range = 100.0;
  bool   extrinsic_est_en = false;
  std::vector<double> extrinsic_T = { -0.011, -0.02329, 0.04412 };
  std::vector<double> extrinsic_R = { 1, 0, 0, 0, 1, 0, 0, 0, 1 };
  int    max_iteration = 4;
  double filter_size_corner = 0.5;
  double filter_size_surf = 0.5;
  double filter_size_map = 0.5;
  double cube_side_length = 200.0;
  std::string map_file_path;

  // --- publish ---
  bool path_en = false;
  bool scan_publish_en = true;
  bool dense_publish_en = true;
  bool scan_bodyframe_pub_en = true;

  // --- pcd_save ---
  bool pcd_save_en = true;
  int  pcd_save_interval = -1;

  // --- debug ---
  bool runtime_pos_log_enable = false;

  // --- output ---
  std::string output_dir = "./fastlio_output/";

  /// Load from YAML node (the root of the config file)
  static FastLioConfig FromYaml(const YAML::Node& config) {
    FastLioConfig cfg;

    if (!config.IsDefined()) return cfg;

    // Helper macro for safe yaml field reading
    #define TRY_READ(field, node, key, type) \
      try { if (node[key]) cfg.field = node[key].as<type>(); } catch(...) {}

    // common
    if (auto n = config["common"]) {
      TRY_READ(lid_topic, n, "lid_topic", std::string);
      TRY_READ(imu_topic, n, "imu_topic", std::string);
      TRY_READ(time_sync_en, n, "time_sync_en", bool);
      TRY_READ(time_offset_lidar_to_imu, n, "time_offset_lidar_to_imu", double);
    }

    // preprocess
    if (auto n = config["preprocess"]) {
      TRY_READ(lidar_type, n, "lidar_type", int);
      TRY_READ(scan_line, n, "scan_line", int);
      TRY_READ(blind, n, "blind", double);
    }

    // mapping
    if (auto n = config["mapping"]) {
      TRY_READ(acc_cov, n, "acc_cov", double);
      TRY_READ(gyr_cov, n, "gyr_cov", double);
      TRY_READ(b_acc_cov, n, "b_acc_cov", double);
      TRY_READ(b_gyr_cov, n, "b_gyr_cov", double);
      TRY_READ(fov_degree, n, "fov_degree", double);
      TRY_READ(det_range, n, "det_range", double);
      TRY_READ(extrinsic_est_en, n, "extrinsic_est_en", bool);
      TRY_READ(extrinsic_T, n, "extrinsic_T", std::vector<double>);
      TRY_READ(extrinsic_R, n, "extrinsic_R", std::vector<double>);
    }

    // publish
    if (auto n = config["publish"]) {
      TRY_READ(path_en, n, "path_en", bool);
      TRY_READ(scan_publish_en, n, "scan_publish_en", bool);
      TRY_READ(dense_publish_en, n, "dense_publish_en", bool);
      TRY_READ(scan_bodyframe_pub_en, n, "scan_bodyframe_pub_en", bool);
    }

    // pcd_save
    if (auto n = config["pcd_save"]) {
      TRY_READ(pcd_save_en, n, "pcd_save_en", bool);
      TRY_READ(pcd_save_interval, n, "interval", int);
    }

    #undef TRY_READ
    return cfg;
  }

  /// Load from YAML file path
  static FastLioConfig FromYamlFile(const std::string& path) {
    try {
      YAML::Node config = YAML::LoadFile(path);
      return FromYaml(config);
    } catch (const std::exception& e) {
      fprintf(stderr, "Warning: Could not load config %s: %s\n", path.c_str(), e.what());
      fprintf(stderr, "Using default configuration\n");
      return FastLioConfig();
    }
  }
};

}  // namespace fastlio

#endif  // FASTLIO_CONFIG_H
