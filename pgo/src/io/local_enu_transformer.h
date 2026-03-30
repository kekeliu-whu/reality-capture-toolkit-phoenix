#pragma once

#include <Eigen/Core>
#include <proj.h>
#include <string>

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
  LocalENUTransformer(double origin_lat_deg, double origin_lon_deg);
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
  const std::string &GetProj4String() const { return proj4_string_; }

 private:
  double origin_lat_;
  double origin_lon_;
  std::string proj_path_;  // used for proj_ functions
  std::string proj4_string_;
  PJ_CONTEXT *ctx_ = nullptr;
  PJ *transformer_ = nullptr;
};
