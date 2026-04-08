#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rtk_agent/PVTSLNMsg.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <sstream>

#include "migration/proto_io.h"
#include "proto/sensors.pb.h"

DEFINE_string(project_dir, "D:\\ProjectX\\project-3d\\data\\manifold-tech-calib\\MT20260326-162323", "Project directory");
DEFINE_string(output_dir, "D:\\output", "Output dir to save converted data");

// Helper function to convert DDMM.MMMMMM format to decimal degrees
double ConvertGprsDegrees(double gps_coord, char direction) {
  int degrees            = static_cast<int>(gps_coord / 100.0);
  double minutes         = gps_coord - degrees * 100.0;
  double decimal_degrees = degrees + minutes / 60.0;

  if (direction == 'S' || direction == 'W') {
    decimal_degrees = -decimal_degrees;
  }
  return decimal_degrees;
}

// Helper function to parse GPS raw messages from gps.txt
// Expects NMEA format: timestamp,$GNGGA,time,lat,N/S,lon,E/W,fix_quality,num_sats,hdop,altitude,M,geoid_height,M,...
void ParseGpsRawMessages(const std::string& gps_file_path, proto::GpsMsgList& gnss_msg_list, std::ofstream& gnss_csv) {
  std::ifstream gps_file(gps_file_path);
  if (!gps_file.is_open()) {
    spdlog::error("Failed to open GPS file: {}", gps_file_path);
    return;
  }

  std::string line;
  int gga_count = 0;

  while (std::getline(gps_file, line)) {
    // Skip empty lines
    if (line.empty()) {
      continue;
    }

    // Parse NMEA format: timestamp,$GNGGA,time,lat,N/S,lon,E/W,fix_quality,num_sats,hdop,altitude,M,geoid_height,M,...
    std::istringstream iss(line);
    std::string token;
    std::vector<std::string> fields;

    // Split the line by comma
    while (std::getline(iss, token, ',')) {
      fields.push_back(token);
    }

    // Process only GNGGA messages (which contain position and altitude)
    if (fields.size() < 12) {
      continue;  // Not enough fields
    }

    if (fields[1] != "$GNGGA") {
      continue;  // Skip non-GNGGA messages
    }

    try {
      // Extract fields: timestamp,msg_type,time,lat,N/S,lon,E/W,fix_quality,num_sats,hdop,altitude,altitude_unit,...
      double timestamp = std::stod(fields[0]);

      // Skip if fix quality is 0 (no fix)
      int fix_quality = std::stoi(fields[7]);
      if (fix_quality == 0) {
        continue;
      }

      double lat_raw  = std::stod(fields[3]);
      char lat_dir    = fields[4][0];
      double latitude = ConvertGprsDegrees(lat_raw, lat_dir);

      double lon_raw   = std::stod(fields[5]);
      char lon_dir     = fields[6][0];
      double longitude = ConvertGprsDegrees(lon_raw, lon_dir);

      double hdop            = std::stod(fields[9]);
      double altitude        = std::stod(fields[10]);
      std::string alt_unit   = (fields.size() > 11 ? fields[11] : "");
      double geoid_height    = std::stod(fields[12]);
      std::string geoid_unit = (fields.size() > 13 ? fields[13] : "");

      // Save full GNSS row to CSV (all gngga fields)
      double ellipsoid_height = altitude + geoid_height;
      gnss_csv << std::fixed << std::setprecision(6) << timestamp << "," << fields[2] << "," << std::setprecision(8) << latitude << "," << lat_dir
               << "," << std::setprecision(8) << longitude << "," << lon_dir << "," << fix_quality << "," << std::setprecision(0)
               << std::stoi(fields[8]) << "," << std::setprecision(2) << hdop << "," << std::setprecision(6) << altitude << "," << alt_unit << ","
               << std::setprecision(6) << geoid_height << "," << geoid_unit << "," << std::setprecision(6) << ellipsoid_height << ","
               << "\"" << line << "\"\n";

      gga_count++;
      if (gga_count % 100 == 0) {
        spdlog::info("Parsed {} GNGGA messages", gga_count);
      }

      spdlog::info("Parsed GPS message at time: {:.6f} lat={:.8f}, lon={:.8f}, alt={:.3f}, fix_quality={}", timestamp, latitude, longitude, altitude,
                   fix_quality);

      if (fix_quality != 4) {
        continue;
      }

      // Create GPS message if fix quality is 4 (RTK fixed)
      auto new_msg = gnss_msg_list.add_gps_msgs();
      new_msg->set_timestamp(timestamp);
      new_msg->set_latitude(latitude);
      new_msg->set_longitude(longitude);
      new_msg->set_altitude(ellipsoid_height);
      new_msg->set_lat_std(0.05);  // RTK fixed is typically around 5cm accuracy
      new_msg->set_lon_std(0.05);
      new_msg->set_alt_std(0.1);  // Altitude accuracy is typically worse
    } catch (const std::exception& e) {
      spdlog::warn("Failed to parse GPS line: {}, error: {}", line, e.what());
      continue;
    }
  }

  gps_file.close();
  spdlog::info("Loaded {} GNGGA messages from {}", gga_count, gps_file_path);
}

void PointCloudCallback(const livox_ros_driver::CustomMsgConstPtr& msg, std::shared_ptr<proto::LidarMsg>& lidar_msg) {
  for (size_t i = 0; i < msg->points.size(); ++i) {
    auto& pt = msg->points[i];

    if ((pt.tag & 0x30) == 0x10 || (pt.tag & 0x30) == 0x00) {
      auto point = lidar_msg->add_points();
      point->set_x(pt.x);
      point->set_y(pt.y);
      point->set_z(pt.z);
      point->set_intensity(pt.reflectivity);
      point->set_timestamp((pt.offset_time + msg->timebase) * 1e-9);
    }
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (!std::filesystem::exists(FLAGS_output_dir)) {
    spdlog::info("Output directory does not exist. Creating: {}", FLAGS_output_dir);
    std::filesystem::create_directories(FLAGS_output_dir);
  }

  proto::ImuMsgList imu_msg_list;
  proto::GpsMsgList gnss_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  // Find all bag files in project_dir with _{n} suffix
  std::vector<std::string> bag_files;
  if (std::filesystem::exists(FLAGS_project_dir)) {
    for (const auto& entry : std::filesystem::directory_iterator(FLAGS_project_dir)) {
      if (entry.is_regular_file()) {
        const auto& path = entry.path();
        if (path.extension() == ".bag") {
          const auto& filename = path.filename().string();
          // Check if file matches pattern with _{n} suffix before .bag
          if (filename.find("_") != std::string::npos) {
            bag_files.push_back(path.string());
          }
        }
      }
    }
    // Sort bag files to ensure consistent processing order
    std::sort(bag_files.begin(), bag_files.end());
    spdlog::info("Found {} bag files in {}", bag_files.size(), FLAGS_project_dir);
  } else {
    spdlog::error("Project directory does not exist: {}", FLAGS_project_dir);
    return 1;
  }

  // Open GNSS CSV file for writing all fields (NMEA GNGGA rows)
  std::ofstream gnss_csv(FLAGS_output_dir + "/gnss.full.csv");
  gnss_csv << "timestamp,gps_time,latitude,lat_dir,longitude,lon_dir,fix_quality,num_sats,hdop,altitude,altitude_unit,geoid_height,geoid_unit,"
              "ellipsoid_height,raw_line\n";

  // Process all bag files
  for (const auto& bag_file_path : bag_files) {
    spdlog::info("Processing bag file: {}", bag_file_path);

    rosbag::Bag bag;
    try {
      bag.open(bag_file_path, rosbag::bagmode::Read);
    } catch (const std::exception& e) {
      spdlog::error("Failed to open bag file {}: {}", bag_file_path, e.what());
      continue;
    }

    for (const auto& m : rosbag::View(bag)) {
      if (m.getTopic() == "/livox/imu") {
        auto msg = m.instantiate<sensor_msgs::Imu>();

        auto new_msg = imu_msg_list.add_imu_msgs();
        new_msg->set_timestamp(msg->header.stamp.toSec());
        new_msg->set_gx(msg->angular_velocity.x);
        new_msg->set_gy(msg->angular_velocity.y);
        new_msg->set_gz(msg->angular_velocity.z);
        new_msg->set_ax(msg->linear_acceleration.x * 9.8);
        new_msg->set_ay(msg->linear_acceleration.y * 9.8);
        new_msg->set_az(msg->linear_acceleration.z * 9.8);

        // spdlog::info("Processed IMU message at time: {:.6f} acc: {:.3f}, {:.3f}, {:.3f} gyro: {:.3f}, {:.3f}, {:.3f}", msg->header.stamp.toSec(),
        //              msg->linear_acceleration.x * 9.8, msg->linear_acceleration.y * 9.8, msg->linear_acceleration.z * 9.8, msg->angular_velocity.x,
        //              msg->angular_velocity.y, msg->angular_velocity.z);
      } else if (m.getTopic() == "/livox/lidar") {
        auto msg = m.instantiate<livox_ros_driver::CustomMsg>();

        std::shared_ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg};
        PointCloudCallback(msg, lidar_msg);
        lidar_writer.Write(lidar_msg);
      }
    }

    bag.close();
  }

  // Load GPS data from gps/gps.txt file
  std::string gps_file = FLAGS_project_dir + "/gps/gps.txt";
  if (std::filesystem::exists(gps_file)) {
    spdlog::info("Loading GPS data from: {}", gps_file);
    ParseGpsRawMessages(gps_file, gnss_msg_list, gnss_csv);

    gnss_csv.close();
    spdlog::info("GNSS raw data CSV written to: {}/gnss.full.csv", FLAGS_output_dir);
  } else {
    spdlog::warn("GPS file not found at: {}", gps_file);
  }

  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);
  WriteGnssFile(FLAGS_output_dir + "/gnss.dat", gnss_msg_list);

  spdlog::info("done.");
}