#include <glog/logging.h>
#include <sqlite3.h>
#include <boost/filesystem.hpp>
#include <cstring>
#include <iostream>
#include <vector>

DEFINE_string(database_filename, "D:/project_3d/reality-capture-toolkit/.cache/output_dir/xsfm.db", "Database filename");

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
    DLOG(INFO) << "Failed to retrieve camera_id: " << sqlite3_errmsg(db);
  }

  sqlite3_finalize(stmt);
  return camera_ids;
}

// Read parameters of the specified camera
bool readCamera(sqlite3* db, int camera_id, int& width, int& height, std::vector<unsigned char>& params) {
  const char* query  = "SELECT width, height, params FROM cameras WHERE camera_id = ?";
  sqlite3_stmt* stmt = nullptr;

  if (sqlite3_prepare_v2(db, query, -1, &stmt, nullptr) != SQLITE_OK) {
    DLOG(INFO) << "Preparation for reading failed: " << sqlite3_errmsg(db);
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

  DLOG(INFO) << "camera_id " << camera_id << " not found.";
  sqlite3_finalize(stmt);
  return false;
}

// Update the 'params' field
bool updateParams(sqlite3* db, int camera_id, const std::vector<unsigned char>& new_params) {
  const char* sql    = "UPDATE cameras SET params = ? WHERE camera_id = ?";
  sqlite3_stmt* stmt = nullptr;

  if (sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr) != SQLITE_OK) {
    DLOG(INFO) << "Preparation for update failed: " << sqlite3_errmsg(db);
    return false;
  }

  sqlite3_bind_blob(stmt, 1, new_params.data(), new_params.size(), SQLITE_STATIC);
  sqlite3_bind_int(stmt, 2, camera_id);

  bool success = sqlite3_step(stmt) == SQLITE_DONE;
  if (!success) {
    DLOG(INFO) << "Update failed: " << sqlite3_errmsg(db);
  }

  sqlite3_finalize(stmt);
  return success;
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  CHECK(boost::filesystem::is_regular_file(FLAGS_database_filename));

  sqlite3* db = nullptr;
  if (sqlite3_open(FLAGS_database_filename.c_str(), &db) != SQLITE_OK) {
    DLOG(INFO) << "Failed to open database: " << sqlite3_errmsg(db);
    return 1;
  }

  // Retrieve all camera IDs
  std::vector<int> camera_ids = getAllCameraIds(db);

  // Iterate over each camera and update its 'params'
  for (int cam_id : camera_ids) {
    int width = 0, height = 0;
    std::vector<unsigned char> params;

    if (readCamera(db, cam_id, width, height, params)) {
      DLOG(INFO) << "Camera " << cam_id << " (" << width << "x" << height << "), original parameter length: " << params.size();

      CHECK(params.size() % sizeof(double) == 0) << params.size();
      CHECK_GT(params.size(), 4 * sizeof(double));

      // Replace with new parameters as needed; here is an example:
      ((double*)params.data())[2] = width / 2;
      ((double*)params.data())[3] = height / 2;

      if (updateParams(db, cam_id, params)) {
        DLOG(INFO) << "Camera " << cam_id << "'s params has been updated.";
      }
    }
  }

  sqlite3_close(db);

  DLOG(INFO) << "done.";
  std::cout << "done." << std::endl;

  return 0;
}