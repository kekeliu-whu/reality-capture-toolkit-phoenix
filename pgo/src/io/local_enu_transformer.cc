#include "io/local_enu_transformer.h"

#include <iomanip>
#include <sstream>

#include <spdlog/spdlog.h>

LocalENUTransformer::LocalENUTransformer(double origin_lat_deg,
                                         double origin_lon_deg)
    : origin_lat_(origin_lat_deg),
      origin_lon_(origin_lon_deg) {
  ctx_ = proj_context_create();

  // Build target local ENU projection (tmerc centered at origin)
  std::ostringstream oss;
  oss << std::fixed << std::setprecision(9)
      << "+proj=tmerc +lat_0=" << origin_lat_deg
      << " +lon_0=" << origin_lon_deg
      << " +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
  std::string target_proj_str = oss.str();
  proj4_string_ = target_proj_str;

  // Source CRS: WGS84 geographic (EPSG:4326)
  transformer_ = proj_create_crs_to_crs(ctx_, "EPSG:4326", target_proj_str.c_str(), nullptr);
  if (!transformer_) {
    spdlog::error("LocalENUTransformer: failed to create PROJ transformer to {}", target_proj_str);
    exit(1);
  }
  spdlog::info("LocalENUTransformer initialized: origin lat={:.6f}deg, lon={:.6f}deg",
               origin_lat_deg, origin_lon_deg);
}

LocalENUTransformer::~LocalENUTransformer() {
  if (transformer_) {
    proj_destroy(transformer_);
    transformer_ = nullptr;
  }
  if (ctx_) {
    proj_context_destroy(ctx_);
    ctx_ = nullptr;
  }
}

Eigen::Vector3d LocalENUTransformer::Convert(double lat_deg, double lon_deg,
                                              double alt_m) const {
  // EPSG:4326 expects (latitude, longitude) order
  PJ_COORD coord_in = proj_coord(lat_deg, lon_deg, alt_m, 0);
  PJ_COORD coord_out = proj_trans(transformer_, PJ_FWD, coord_in);

  if (coord_out.xyz.x == HUGE_VAL || coord_out.xyz.y == HUGE_VAL) {
    spdlog::warn("LocalENUTransformer::Convert failed for lat={:.6f}, lon={:.6f}", lat_deg, lon_deg);
    return Eigen::Vector3d::Zero();
  }
  // tmerc output: x=East, y=North; altitude delta as Up
  return Eigen::Vector3d(coord_out.xyz.x, coord_out.xyz.y, alt_m);
}
