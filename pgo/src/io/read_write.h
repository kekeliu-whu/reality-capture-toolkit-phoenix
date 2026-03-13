#pragma once

#include <string>
#include <vector>

#include "common/types.h"

/**
 * @brief Load the submaps from a las file.
 *
 * The las file will contain all the points in the submaps. The timestamp of the
 * submap will be stored in the time field of the las file.
 *
 * @param project
 * @param scans
 * @param submap_duration_secs <=0 means do not merge submaps
 */
void LoadSubmapList(const std::string &project,
                    std::vector<TimestampedPointCloud> &scans,
                    double submap_duration_secs);

/**
 * @brief Save the submaps to a las file.
 *
 * The las file will contain all the points in the submaps. The timestamp of the
 * submap will be stored in the time field of the las file.
 *
 * @param submaps
 * @param output_filename
 */
void SaveLasFile(const std::vector<TimestampedPointCloud> &submaps,
                 const std::string &output_filename);

/**
 * @brief Load GNSS/RTK data from gnss.dat file
 *
 * The gnss.dat file should contain GNSS measurements with format:
 * timestamp latitude longitude altitude lat_std lon_std alt_std
 * [optional RTK fields: fix_type num_satellites gdop hdop vdop baseline heading pitch roll heading_std]
 *
 * @param gnss_filename Path to gnss.dat file
 * @param gnss_data Vector to store loaded GNSS data
 * @return true if successfully loaded, false otherwise
 */
bool LoadGnssData(const std::string &gnss_filename,
                  std::vector<GpsData> &gnss_data);

/**
 * @brief Load GNSS/RTK data from protobuf format
 *
 * @param project_path Path to project directory
 * @param gnss_data Vector to store loaded GNSS data
 * @return true if successfully loaded, false otherwise
 */
bool LoadGnssDataFromProject(const std::string &project_path,
                             std::vector<GpsData> &gnss_data);

#include <proj.h>

/**
 * @brief LocalENU coordinate transformer using PROJ library
 *
 * Converts from WGS84 (latitude/longitude/altitude) to local ENU
 * (East-North-Up) coordinates. The origin is the first GNSS measurement.
 */
class LocalENUTransformer {
 public:
  /**
   * @brief Initialize transformer with WGS84 origin (lat/lon in degrees)
   */
  LocalENUTransformer(double origin_lat_deg, double origin_lon_deg,
                      double origin_alt_m);
  ~LocalENUTransformer();

  // Non-copyable due to PROJ resource ownership
  LocalENUTransformer(const LocalENUTransformer &) = delete;
  LocalENUTransformer &operator=(const LocalENUTransformer &) = delete;

  /**
   * @brief Convert WGS84 coordinates to local ENU (meters)
   */
  Eigen::Vector3d Convert(double lat_deg, double lon_deg, double alt_m) const;

  double GetOriginLat() const { return origin_lat_; }
  double GetOriginLon() const { return origin_lon_; }

 private:
  double origin_lat_;
  double origin_lon_;
  double origin_alt_;
  PJ_CONTEXT *ctx_ = nullptr;
  PJ *transformer_ = nullptr;
};
