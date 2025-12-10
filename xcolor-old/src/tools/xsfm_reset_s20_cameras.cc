#include <spdlog/spdlog.h>
#include <sqlite3.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cstring>
#include <iostream>
#include <vector>

DEFINE_string(database_filename, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/xsfm.db", "Database filename");
DEFINE_string(calibration_filename, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/calibration.yaml",
              "Calibration filename");

// Retrieve all camera_ids
std::vector<int> getAllCameraIds(sqlite3* db) {
  std::vector<int> camera_ids;
  const char* query  = "SELECT camera_id FROM cameras;";
  sqlite3_stmt* stmt = nullptr;

  if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) == SQLITE_OK) {
    while (sqlite3_step(stmt) == SQLITE_ROW) {
      int cam_id = sqlite3_column_int(stmt, 0);
      camera_ids.push_back(cam_id);
    }
  } else {
    spdlog::debug("Failed to retrieve camera_id: {}", sqlite3_errmsg(db));
  }

  sqlite3_finalize(stmt);
  return camera_ids;
}

// Read parameters of the specified camera
bool readCamera(sqlite3* db, int camera_id, int& width, int& height, std::vector<unsigned char>& params) {
  const char* query  = "SELECT width, height, params FROM cameras WHERE camera_id = ?";
  sqlite3_stmt* stmt = nullptr;

  if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
    spdlog::debug("Preparation for reading failed: {}", sqlite3_errmsg(db));
    return false;
  }

  sqlite3_bind_int(stmt, 1, camera_id);

  if (sqlite3_step(stmt) == SQLITE_ROW) {
    width            = sqlite3_column_int(stmt, 0);
    height           = sqlite3_column_int(stmt, 1);
    const void* blob = sqlite3_column_blob(stmt, 2);
    int blob_size    = sqlite3_column_bytes(stmt, 2);
    params.assign((const unsigned char*)blob, (const unsigned char*)blob + blob_size);
    sqlite3_finalize(stmt);
    return true;
  }

  spdlog::debug("camera_id {} not found.", camera_id);
  sqlite3_finalize(stmt);
  return false;
}

// Update the 'params' field
bool updateParams(sqlite3* db, int camera_id, const std::vector<unsigned char>& new_params) {
  const char* sql    = "UPDATE cameras SET params = ? WHERE camera_id = ?";
  sqlite3_stmt* stmt = nullptr;

  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    spdlog::debug("Preparation for update failed: {}", sqlite3_errmsg(db));
    return false;
  }

  sqlite3_bind_blob(stmt, 1, new_params.data(), new_params.size(), SQLITE_STATIC);
  sqlite3_bind_int(stmt, 2, camera_id);

  bool success = sqlite3_step(stmt) == SQLITE_DONE;
  if (!success) {
    spdlog::debug("Update failed: {}", sqlite3_errmsg(db));
  }

  sqlite3_finalize(stmt);
  return success;
}

bool getFirstRow(const std::string& db_path, std::string& name, int& camera_id) {
  sqlite3* db;
  sqlite3_stmt* stmt;

  int rc = sqlite3_open(db_path.c_str(), &db);
  if (rc != SQLITE_OK) {
    std::cerr << "Error opening database: " << sqlite3_errmsg(db) << std::endl;
    return false;
  }

  const char* sql = "SELECT name, camera_id FROM images LIMIT 1;";

  rc = sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr);
  if (rc != SQLITE_OK) {
    std::cerr << "Error preparing statement: " << sqlite3_errmsg(db) << std::endl;
    sqlite3_close(db);
    return false;
  }

  rc = sqlite3_step(stmt);
  if (rc == SQLITE_ROW) {
    const unsigned char* db_name = sqlite3_column_text(stmt, 0);
    name                         = db_name ? reinterpret_cast<const char*>(db_name) : "";
    camera_id                    = sqlite3_column_int(stmt, 1);

    sqlite3_finalize(stmt);
    sqlite3_close(db);
    return true;
  } else if (rc == SQLITE_DONE) {
    std::cerr << "No data found in the table." << std::endl;
  } else {
    std::cerr << "Error executing statement: " << sqlite3_errmsg(db) << std::endl;
  }

  sqlite3_finalize(stmt);
  sqlite3_close(db);
  return false;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  spdlog::set_level(spdlog::level::debug);

  if (!boost::filesystem::is_regular_file(FLAGS_database_filename)) {
    spdlog::error("Database file not found: {}", FLAGS_database_filename);
    exit(1);
  }

  sqlite3* db = nullptr;
  if (sqlite3_open(FLAGS_database_filename.c_str(), &db) != SQLITE_OK) {
    spdlog::debug("Failed to open database: {}", sqlite3_errmsg(db));
    return 1;
  }

  // Retrieve all camera IDs
  std::vector<int> camera_ids = getAllCameraIds(db);
  if (camera_ids.size() != 2) {
    spdlog::error("Expected 2 camera IDs, got {}", camera_ids.size());
    exit(1);
  }

  std::string image_path;
  int camera_id;
  if (getFirstRow(FLAGS_database_filename, image_path, camera_id)) {
    spdlog::debug("First Row - Name: {}, Camera ID: {}", image_path, camera_id);

    bool is_left = (image_path.find("left") != std::string::npos);

    if ((is_left && camera_ids[1] == camera_id) || (!is_left && camera_ids[0] == camera_id)) {
      std::swap(camera_ids[0], camera_ids[1]);
    }

    spdlog::debug("Sorted Camera IDs: [{}, {}]", camera_ids[0], camera_ids[1]);
  } else {
    spdlog::critical("Failed to retrieve the first row.");
    exit(1);
  }

  std::vector<Eigen::Vector2d> camera_params;
  try {
    if (!boost::filesystem::is_regular_file(FLAGS_calibration_filename)) {
      spdlog::error("Calibration file not found: {}", FLAGS_calibration_filename);
      exit(1);
    }

    YAML::Node config = YAML::LoadFile(FLAGS_calibration_filename);

    YAML::Node left_cam = config["intrinsic"]["fisheye_left"];
    double left_a11     = left_cam["projection_parameters"]["A11"].as<double>();
    double left_a22     = left_cam["projection_parameters"]["A22"].as<double>();
    camera_params.push_back({left_a11, left_a22});

    YAML::Node right_cam = config["intrinsic"]["fisheye_right"];
    double right_a11     = right_cam["projection_parameters"]["A11"].as<double>();
    double right_a22     = right_cam["projection_parameters"]["A22"].as<double>();
    camera_params.push_back({right_a11, right_a22});
  } catch (const YAML::Exception& e) {
    spdlog::error("YAML error: {}", e.what());
    return 1;
  } catch (const std::exception& e) {
    spdlog::error("Error: {}", e.what());
    return 1;
  }

  // Iterate over each camera and update its 'params'
  for (int i = 0; i < camera_ids.size(); ++i) {
    auto cam_id       = camera_ids[i];
    auto camera_param = camera_params[i];

    int width = 0, height = 0;
    std::vector<unsigned char> params;

    if (readCamera(db, cam_id, width, height, params)) {
      spdlog::debug("Camera {} ({}x{}), original parameter length: {}", cam_id, width, height, params.size());

      if (params.size() % sizeof(double) != 0) {
        spdlog::error("Invalid params size: {}", params.size());
        exit(1);
      }
      if (params.size() <= 4 * sizeof(double)) {
        spdlog::error("Params size too small: {}", params.size());
        exit(1);
      }

      // Replace with new parameters as needed; here is an example:
      ((double*)params.data())[0] = camera_param[0];
      ((double*)params.data())[1] = width / 2;
      ((double*)params.data())[2] = height / 2;

      if (updateParams(db, cam_id, params)) {
        spdlog::debug("Camera {}'s params has been updated: fx = {}, fy = {}, cx = {}, cy = {}", cam_id, camera_param[0], camera_param[1], width / 2, height / 2);
      }
    }
  }

  sqlite3_close(db);

  spdlog::debug("done.");
  std::cout << "done." << std::endl;

  return 0;
}