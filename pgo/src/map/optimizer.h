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
 */
void OptimizeWithGnss(std::vector<TimestampedPointCloud> &submaps,
                      const std::vector<GpsData> &gnss_data,
                      const proto::PgoConfig &config,
                      bool use_rtk = false);
