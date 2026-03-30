#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rtk_agent/PVTSLNMsg.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <filesystem>
#include <fstream>
#include <sstream>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(project_dir, "D:\\ProjectX\\project-3d\\data\\manifold-tech-calib\\MT20260326-162323", "Project directory");
DEFINE_string(calib_filename, "D:\\ProjectX\\project-3d\\data\\manifold-tech-calib\\ikalibr_param.yaml", "Calibration filename");
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

template <int Rows, int Cols>
Eigen::Matrix<double, Rows, Cols> ReadMatrix(const std::vector<double>& mat) {
  CHECK_EQ(Rows * Cols, mat.size());
  return Eigen::Map<const Eigen::Matrix<double, Rows, Cols, Eigen::RowMajor>>{mat.data()};
}

template <int Rows, int Cols>
void ReadMatrixToProto(const YAML::Node& node, google::protobuf::RepeatedField<double>* mat_proto) {
  auto mat_vec = node.as<std::vector<double>>();
  CHECK_EQ(Rows * Cols, mat_vec.size());
  for (auto& e : mat_vec) {
    mat_proto->Add(e);
  }
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

  proto::SensorCalib sc;
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

  {
    // Read calibration from YAML file
    try {
      YAML::Node yaml_root    = YAML::LoadFile(FLAGS_calib_filename);
      const auto& calib_param = yaml_root["CalibParam"];
      const auto& extri       = calib_param["EXTRI"];
      const auto& temporal    = calib_param["TEMPORAL"];

      // Read SO3_LkToBr (Lidar rotation relative to body frame)
      try {
        for (const auto& item : extri["SO3_LkToBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            const auto& quat   = item["value"];
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_rx(quat["qx"].as<double>());
            lidar_to_body->set_ry(quat["qy"].as<double>());
            lidar_to_body->set_rz(quat["qz"].as<double>());
            lidar_to_body->set_rw(quat["qw"].as<double>());
            spdlog::info("Loaded Lidar rotation: qx={}, qy={}, qz={}, qw={}", quat["qx"].as<double>(), quat["qy"].as<double>(),
                         quat["qz"].as<double>(), quat["qw"].as<double>());
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read SO3_LkToBr: {}", e.what());
      }

      // Read POS_LkInBr (Lidar position relative to body frame)
      try {
        for (const auto& item : extri["POS_LkInBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            const auto& pos    = item["value"];
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_tx(pos["r0c0"].as<double>());
            lidar_to_body->set_ty(pos["r1c0"].as<double>());
            lidar_to_body->set_tz(pos["r2c0"].as<double>());
            spdlog::info("Loaded Lidar position: tx={}, ty={}, tz={}", pos["r0c0"].as<double>(), pos["r1c0"].as<double>(), pos["r2c0"].as<double>());
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read POS_LkInBr: {}", e.what());
      }

      // Read TO_LkToBr (Lidar time offset relative to body frame)
      try {
        for (const auto& item : temporal["TO_LkToBr"]) {
          if (item["key"].as<std::string>() == "/livox/lidar") {
            double time_offset = item["value"].as<double>();
            auto lidar_to_body = sc.mutable_lidar_to_encoder();
            lidar_to_body->set_time_offset(time_offset);
            spdlog::info("Loaded Lidar time offset: {}", time_offset);
            break;
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read TO_LkToBr: {}", e.what());
      }

      // Read and output camera extrinsic and intrinsic parameters
      try {
        const auto& so3_list    = extri["SO3_CmToBr"];
        const auto& pos_list    = extri["POS_CmInBr"];
        const auto& to_cm_list  = temporal["TO_CmToBr"];
        const auto& camera_list = calib_param["INTRI"]["Camera"];

        for (const auto& so3_item : so3_list) {
          try {
            std::string cam_key = so3_item["key"].as<std::string>();
            spdlog::info("Processing camera: {}", cam_key);

            auto cam_param = sc.add_camera_param();
            cam_param->set_name(cam_key);

            // Find and set matching position entry
            try {
              for (const auto& pos_item : pos_list) {
                if (pos_item["key"].as<std::string>() == cam_key) {
                  const auto& quat = so3_item["value"];
                  const auto& pos  = pos_item["value"];

                  // Set camera to body extrinsic (rotation + translation)
                  auto extrinsic = cam_param->mutable_extrinsic();
                  extrinsic->set_rx(quat["qx"].as<double>());
                  extrinsic->set_ry(quat["qy"].as<double>());
                  extrinsic->set_rz(quat["qz"].as<double>());
                  extrinsic->set_rw(quat["qw"].as<double>());
                  extrinsic->set_tx(pos["r0c0"].as<double>());
                  extrinsic->set_ty(pos["r1c0"].as<double>());
                  extrinsic->set_tz(pos["r2c0"].as<double>());

                  spdlog::info("{} extrinsic: rot({}, {}, {}, {}), trans({}, {}, {})", cam_key, quat["qx"].as<double>(), quat["qy"].as<double>(),
                               quat["qz"].as<double>(), quat["qw"].as<double>(), pos["r0c0"].as<double>(), pos["r1c0"].as<double>(),
                               pos["r2c0"].as<double>());
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read position for camera {}: {}", cam_key, e.what());
            }

            // Find and set camera time offset
            try {
              for (const auto& time_item : to_cm_list) {
                if (time_item["key"].as<std::string>() == cam_key) {
                  double time_offset = time_item["value"].as<double>();
                  cam_param->mutable_extrinsic()->set_time_offset(time_offset);
                  spdlog::info("{} time offset: {}", cam_key, time_offset);
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read time offset for camera {}: {}", cam_key, e.what());
            }

            // Find and set camera intrinsic parameters
            try {
              for (const auto& cam_item : camera_list) {
                if (cam_item["key"].as<std::string>() == cam_key) {
                  const auto& cam_data = cam_item["value"]["ptr_wrapper"]["data"];

                  // Set focal length
                  auto focal = cam_data["focal_length"];
                  cam_param->set_fx(focal[0].as<double>());
                  cam_param->set_fy(focal[1].as<double>());

                  // Set principal point
                  auto principal = cam_data["principal_point"];
                  cam_param->set_cx(principal[0].as<double>());
                  cam_param->set_cy(principal[1].as<double>());

                  // Set distortion parameters
                  auto disto = cam_data["disto_param"];
                  cam_param->set_k1(disto[0].as<double>());
                  cam_param->set_k2(disto[1].as<double>());
                  cam_param->set_k3(disto[2].as<double>());
                  cam_param->set_k4(disto[3].as<double>());

                  spdlog::info("{} intrinsic: fx={}, fy={}, cx={}, cy={}, k1={}, k2={}, k3={}, k4={}", cam_key, cam_param->fx(), cam_param->fy(),
                               cam_param->cx(), cam_param->cy(), cam_param->k1(), cam_param->k2(), cam_param->k3(), cam_param->k4());
                  break;
                }
              }
            } catch (const std::exception& e) {
              spdlog::warn("Failed to read intrinsic for camera {}: {}", cam_key, e.what());
            }
          } catch (const std::exception& e) {
            spdlog::warn("Failed to process camera: {}", e.what());
          }
        }
      } catch (const std::exception& e) {
        spdlog::warn("Failed to read camera extrinsic and intrinsic parameters: {}", e.what());
      }

      WriteSensorCalibFile(FLAGS_output_dir + "/calibration.dat", sc);
      spdlog::info("Calibration file saved to: {}/calibration.dat", FLAGS_output_dir);
    } catch (const std::exception& e) {
      spdlog::error("Failed to load calibration file: {}", e.what());
    }
  }

  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);
  WriteGnssFile(FLAGS_output_dir + "/gnss.dat", gnss_msg_list);

  spdlog::info("done.");
}