#pragma once

#include <string>
#include <vector>

#include "common/types.h"
#include "io/local_enu_transformer.h"

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
 * @param proj4_string Optional PROJ4 coordinate system string (e.g. LocalENU)
 */
void SaveLasFile(const std::vector<TimestampedPointCloud> &submaps,
                 const std::string &output_filename,
                 const std::string &proj4_string = "");

/**
 * @brief Load GNSS/RTK data from protobuf format
 *
 * @param project_path Path to project directory
 * @param gnss_data Vector to store loaded GNSS data
 * @return true if successfully loaded, false otherwise
 */
bool LoadGnssDataFromProject(const std::string &project_path,
                             std::vector<GpsData> &gnss_data);
