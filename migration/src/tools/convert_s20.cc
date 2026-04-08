#include <gflags/gflags.h>
#include <livox_ros_driver/CustomMsg.h>
#include <nav_msgs/Odometry.h>
#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <rtk_agent/PVTSLNMsg.h>
#include <sensor_msgs/Imu.h>
#include <sensor_msgs/NavSatFix.h>
#include <filesystem>
#include <fstream>
#include <iomanip>

#include "migration/proto_io.h"
#include "proto/sensors.pb.h"

DEFINE_string(bag_filename, "D:\\Users\\rick\\Desktop\\2026-03-16_09-34-40-with-rtk\\all_2026-03-16-09-34-52.bag", "Point cloud filename");
DEFINE_string(output_dir, "D:\\slam", "Output dir to save converted data");

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

  rosbag::Bag bag;
  bag.open(FLAGS_bag_filename, rosbag::bagmode::Read);

  if (!std::filesystem::exists(FLAGS_output_dir)) {
    spdlog::info("Output directory does not exist. Creating: {}", FLAGS_output_dir);
    std::filesystem::create_directories(FLAGS_output_dir);
  }

  proto::ImuMsgList imu_msg_list;
  proto::GpsMsgList gnss_msg_list;
  SequentialLidarFileWriter<proto::LidarMsg> lidar_writer;
  lidar_writer.Open(FLAGS_output_dir + "/lidar.dat");

  // Open GNSS CSV file for writing all fields
  std::ofstream gnss_csv(FLAGS_output_dir + "/gnss.full.csv");
  gnss_csv << "timestamp,"
           << "bestpos_type,"
           << "bestpos_lat,bestpos_lon,bestpos_hgt,"
           << "bestpos_latstd,bestpos_lonstd,bestpos_hgtstd,"
           << "bestpos_diffage,bestpos_svs,bestpos_solnsvs,"
           << "psrpos_type,psrpos_lat,psrpos_lon,psrpos_hgt,psrpos_svs,psrpos_solnsvs,"
           << "undulation,psrvel_east,psrvel_north,psrvel_ground,"
           << "heading_type,heading_length,heading_degree,heading_pitch,"
           << "heading_trackedsvs,heading_solnsvs,heading_ggl1,heading_ggl1l2,"
           << "gdop,pdop,hdop,htdop,tdop,cutoff,num_sat\n";

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
    } else if (m.getTopic() == "/rtk_agent/pvtsln_sync") {
      auto msg = m.instantiate<rtk_agent::PVTSLNMsg>();

      double timestamp = msg->header.stamp.toSec();

      // Write all fields to CSV
      gnss_csv << std::fixed << std::setprecision(6) << timestamp << "," << (int)msg->bestpos_type.type << "," << std::setprecision(8)
               << msg->bestpos_lat << "," << msg->bestpos_lon << "," << std::setprecision(6) << msg->bestpos_hgt << "," << msg->bestpos_latstd << ","
               << msg->bestpos_lonstd << "," << msg->bestpos_hgtstd << "," << msg->bestpos_diffage << "," << (int)msg->bestpos_svs << ","
               << (int)msg->bestpos_solnsvs << "," << (int)msg->psrpos_type.type << "," << std::setprecision(8) << msg->psrpos_lat << ","
               << msg->psrpos_lon << "," << std::setprecision(6) << msg->psrpos_hgt << "," << (int)msg->psrpos_svs << "," << (int)msg->psrpos_solnsvs
               << "," << msg->undulation << "," << msg->psrvel_east << "," << msg->psrvel_north << "," << msg->psrvel_ground << ","
               << (int)msg->heading_type.type << "," << msg->heading_length << "," << msg->heading_degree << "," << msg->heading_pitch << ","
               << (int)msg->heading_trackedsvs << "," << (int)msg->heading_solnsvs << "," << (int)msg->heading_ggl1 << "," << (int)msg->heading_ggl1l2
               << "," << msg->gdop << "," << msg->pdop << "," << msg->hdop << "," << msg->htdop << "," << msg->tdop << "," << msg->cutoff << ","
               << (int)msg->num_sat << "\n";

      // Only use RTK NARROW_INT fixed solutions
      if (msg->bestpos_type.type != rtk_agent::PosType::NARROW_INT) {
        continue;
      }

      auto new_msg = gnss_msg_list.add_gps_msgs();
      new_msg->set_timestamp(timestamp);
      new_msg->set_latitude(msg->bestpos_lat);
      new_msg->set_longitude(msg->bestpos_lon);
      new_msg->set_altitude(msg->bestpos_hgt);

      // Standard deviations
      new_msg->set_lat_std(msg->bestpos_latstd);
      new_msg->set_lon_std(msg->bestpos_lonstd);
      new_msg->set_alt_std(msg->bestpos_hgtstd);

      // spdlog::info("Processed GNSS message at time: {:.6f} lat={:.8f}, lon={:.8f}, alt={:.3f}",
      //              timestamp, msg->bestpos_lat, msg->bestpos_lon, msg->bestpos_hgt);
    }
  }

  // Close GNSS CSV file
  gnss_csv.close();
  spdlog::info("GNSS raw data CSV written to: {}/gnss_raw.csv", FLAGS_output_dir);

  WriteImuFile(FLAGS_output_dir + "/imu.dat", imu_msg_list);
  WriteGnssFile(FLAGS_output_dir + "/gnss.dat", gnss_msg_list);

  spdlog::info("done.");
}