#include <custom_msgs/LixelAnyData.h>
#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <sensor_msgs/Imu.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

#include "migration/proto_io.h"
#include "proto/calib.pb.h"
#include "proto/sensors.pb.h"

DEFINE_string(bag_filename, "D:/ProjectX/project-3d/data/sfm/mixed/indoor-office/2026-01-14_15-29-47/all_2026-01-14-15-29-55.bag",
              "Point cloud filename");
DEFINE_string(output_dir, "D:/ProjectX/project-3d/data/sfm/mixed/indoor-office/2026-01-14_15-29-47/slam", "Output dir to save converted data");

#pragma pack(push, 1)
struct Point {
  double timestamp;
  float x;
  float y;
  float z;
  uint8_t intensity;
  uint8_t channel_num;
  uint8_t echo_num;
  uint8_t rest;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct ScanData {
  uint64_t point_num;
  Point *points;
};
#pragma pack(pop)

#pragma pack(push, 1)
struct LidarScan {
  uint64_t scan_data_len;
  ScanData scan_data;
};
#pragma pack(pop)

// LiDAR file reader class
class LidarFileReader {
 private:
  std::ifstream file_;
  std::string filename_;
  double last_frame_time_;
  double frame_interval_;
  bool is_initialized_;
  bool file_ended_;
  uint64_t total_points;
  uint64_t read_points;

 public:
  LidarFileReader(const std::string &filename, double frame_interval = 0.1)
      : filename_(filename),
        last_frame_time_(0.0),
        frame_interval_(frame_interval),
        is_initialized_(false),
        file_ended_(false),
        total_points(0),
        read_points(0) {
    if (!boost::filesystem::exists(filename)) {
      spdlog::critical("LiDAR file does not exist: {}", filename);
      exit(1);
    }

    file_ = std::ifstream(filename.c_str(), std::ios::binary);
    if (!file_) {
      spdlog::critical("Failed to open LiDAR file: {}", filename);
      exit(1);
    }

    // Skip header line
    int64_t scan_data_len;
    file_.read(reinterpret_cast<char *>(&scan_data_len), sizeof(uint64_t));
    int64_t point_num;
    file_.read(reinterpret_cast<char *>(&point_num), sizeof(uint64_t));
    total_points = point_num;

    spdlog::info("Opened LiDAR file: {}", filename);
  }

  // Read one frame of point cloud data
  // Return value: true if successfully read, false if file ended
  bool readOneScan(livox_ros_driver::CustomMsg::Ptr &scan_msg) {
    if (file_ended_ || !file_) {
      return false;
    }

    scan_msg.reset(new livox_ros_driver::CustomMsg());
    scan_msg->header.frame_id = "lidar_link";
    scan_msg->point_num       = 0;

    while (true) {
      Point pt;
      if (!file_.read(reinterpret_cast<char *>(&pt), sizeof(Point))) {
        file_ended_ = true;
        return false;
      }
      read_points++;

      // Initialize first frame
      if (!is_initialized_) {
        last_frame_time_       = pt.timestamp;
        scan_msg->header.stamp = ros::Time(pt.timestamp);
        scan_msg->timebase     = static_cast<uint64_t>(pt.timestamp * 1e9);
        is_initialized_        = true;
      }

      // Check if current frame time range is exceeded
      if (pt.timestamp - last_frame_time_ >= frame_interval_ && scan_msg->point_num > 0) {
        // Current line is the first point of next frame, need to roll back and process in next call
        // Since file pointer cannot be rolled back to line start, we need to save this point
        // For simplified processing, add this point to next frame
        // Update frame time to this point's time
        last_frame_time_ = pt.timestamp;

        spdlog::info("Read a scan with {} points.", scan_msg->point_num);
        // Return current frame (excluding this new point)
        return scan_msg->point_num > 0;
      }

      // If it's the start of a new frame (point count is 0 and initialized)
      if (scan_msg->point_num == 0 && is_initialized_) {
        scan_msg->header.stamp = ros::Time(last_frame_time_);
        scan_msg->timebase     = static_cast<uint64_t>(last_frame_time_ * 1e9);
      }

      // Add point to current frame
      livox_ros_driver::CustomPoint point;
      point.x            = pt.x;
      point.y            = pt.y;
      point.z            = pt.z;
      point.reflectivity = static_cast<uint8_t>(pt.intensity);
      point.offset_time  = static_cast<uint32_t>((pt.timestamp - scan_msg->header.stamp.toSec()) * 1e9);
      point.line         = pt.channel_num;
      point.tag          = 0;

      scan_msg->points.push_back(point);
      scan_msg->point_num++;
    }

    // File ended, return last frame (if there is data)
    return scan_msg->point_num > 0;
  }

  bool isFileEnded() const { return file_ended_; }

  void setFrameInterval(double interval) { frame_interval_ = interval; }

  double getProgress() const { return total_points > 0 ? (read_points / (double)total_points) * 100.0 : 0.0; }
};

// Load calibration parameters from JSON file
void load_calibration_from_file(const std::string &filename, Eigen::Matrix3d &Lidar_R_wrt_IMU,
                                Eigen::Vector3d &Lidar_T_wrt_IMU, double &time_offset) {
  if (!boost::filesystem::exists(filename)) {
    spdlog::warn("Calibration file does not exist: {}", filename);
    exit(1);
  }

  std::ifstream cal_file(filename);
  if (!cal_file) {
    spdlog::warn("Failed to open calibration file: {}", filename);
    exit(1);
  }

  try {
    nlohmann::json root;
    cal_file >> root;

    // Extract Lidar_offset values (Lidar_T_wrt_IMU)
    if (root.contains("Info") && root["Info"].contains("Lidar_Parameter") && root["Info"]["Lidar_Parameter"].contains("Lidar_to_IMU") &&
        root["Info"]["Lidar_Parameter"]["Lidar_to_IMU"].contains("Lidar_offset")) {
      const auto &offset = root["Info"]["Lidar_Parameter"]["Lidar_to_IMU"]["Lidar_offset"];
      double x           = offset["x"].get<double>();
      double y           = offset["y"].get<double>();
      double z           = offset["z"].get<double>();
      Lidar_T_wrt_IMU    = Eigen::Vector3d(x, y, z);
      spdlog::info("Loaded Lidar_T_wrt_IMU from calibration: [{:.6f}, {:.6f}, {:.6f}]", x, y, z);
    }

    // Extract Lidar_Rotations (quaternion) and convert to rotation matrix (Lidar_R_wrt_IMU)
    if (root.contains("Info") && root["Info"].contains("Lidar_Parameter") && root["Info"]["Lidar_Parameter"].contains("Lidar_to_IMU") &&
        root["Info"]["Lidar_Parameter"]["Lidar_to_IMU"].contains("Lidar_Rotations")) {
      const auto &quat_json = root["Info"]["Lidar_Parameter"]["Lidar_to_IMU"]["Lidar_Rotations"];
      double qw             = quat_json["qw"].get<double>();
      double qx             = quat_json["qx"].get<double>();
      double qy             = quat_json["qy"].get<double>();
      double qz             = quat_json["qz"].get<double>();

      time_offset = root["Info"]["Lidar_Parameter"]["Lidar_to_IMU"]["ImutimeCorrectMs"].get<double>();

      Eigen::Quaterniond quat(qw, qx, qy, qz);
      Lidar_R_wrt_IMU = quat.toRotationMatrix();

      Lidar_T_wrt_IMU = -(Lidar_R_wrt_IMU.transpose() * Lidar_T_wrt_IMU);
      Lidar_R_wrt_IMU.transposeInPlace();

      spdlog::info(
          "Loaded Lidar_R_wrt_IMU (from quaternion) from calibration: qw={:.6f}, qx={:.6f}, qy={:.6f}, qz={:.6f}, "
          "time_offset={:.6f}",
          qw, qx, qy, qz, time_offset);
    }
  } catch (const std::exception &e) {
    time_offset = 0.0;
    Lidar_R_wrt_IMU.setIdentity();
    Lidar_T_wrt_IMU.setZero();
    spdlog::warn("Error parsing calibration file: {}", e.what());
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  rosbag::Bag bag;
  bag.open(FLAGS_bag_filename, rosbag::bagmode::Read);

  proto::SensorCalib sc;
  proto::ImuMsgList imu_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  for (const auto &m : rosbag::View(bag)) {
    if (m.getTopic() == "/livox/imu") {
      auto msg = m.instantiate<sensor_msgs::Imu>();

      auto new_msg = imu_msg_list.add_imu_msgs();
      new_msg->set_timestamp(msg->header.stamp.toSec());
      new_msg->set_gx(msg->angular_velocity.x);
      new_msg->set_gy(msg->angular_velocity.y);
      new_msg->set_gz(msg->angular_velocity.z);
      new_msg->set_ax(msg->linear_acceleration.x);
      new_msg->set_ay(msg->linear_acceleration.y);
      new_msg->set_az(msg->linear_acceleration.z);
    } else if (m.getTopic() == "/livox/lidar") {
      auto msg = m.instantiate<livox_ros_driver::CustomMsg>();

      std::shared_ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg};
      // PointCloudCallback(msg, lidar_msg);
      lidar_writer.Write(lidar_msg);
    }
  }
  WriteSensorCalibFile(FLAGS_output_dir + "/calibration.dat", sc);
  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);

  spdlog::info("done.");
}