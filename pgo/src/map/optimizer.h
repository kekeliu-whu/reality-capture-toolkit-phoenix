#pragma once

#include "common/types.h"
#include "proto/sensors.pb.h"

void Optimize(std::vector<TimestampedPointCloud> &submaps,
              const proto::PgoConfig &config);

/**
 * @brief Optimize with GNSS/RTK data fusion
 * 
 * The first GNSS measurement's position is automatically used as the origin
 * for LocalENU coordinate system conversion.
 * 
 * @param submaps Vector of submaps to optimize
 * @param gnss_data Vector of GNSS/RTK measurements
 * @param config PGO configuration
 * @param use_rtk Whether to enable RTK constraints
 * @param use_btc Whether to enable BTC-based loop closure constraints
 * @param proj4_string Output reference to receive PROJ4 coordinate system string
 */
void OptimizeWithGnss(std::vector<TimestampedPointCloud> &submaps,
                      const std::vector<GpsData> &gnss_data,
                      const proto::PgoConfig &config,
                      bool use_rtk,
                      bool use_btc,
                      std::string &proj4_string);

/**
 * @brief Add BTC (Binary Triangle Cluster) based loop closure constraints
 * 
 * Uses the external BTC library to detect loop closures based on structural
 * descriptors. Only adds constraints between submaps with large time differences
 * but small spatial distances.
 * 
 * @param problem Ceres optimization problem
 * @param submaps Vector of submaps with point clouds
 * @param config PGO configuration
 */
void AddBTCConstraints(ceres::Problem &problem,
                       std::vector<TimestampedPointCloud> &submaps,
                       const proto::PgoConfig &config);
