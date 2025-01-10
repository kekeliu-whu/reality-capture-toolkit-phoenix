#pragma once

#include <string>

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
