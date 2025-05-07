#include <gtest/gtest.h>

#include "migration/proto_io.h"

TEST(SensorIO, ReadWriteImuFile) {
  std::string filename = "/tmp/imu.dat";

  proto::ImuMsgList imu;
  auto it = imu.add_imu_msgs();
  it->set_timestamp(1000);
  it->set_ax(0.1);
  it->set_ay(0.2);
  it->set_az(0.3);
  it->set_gx(0.4);
  it->set_gy(0.5);
  it->set_gz(0.6);
  it = imu.add_imu_msgs();
  it->set_timestamp(1000);
  it->set_ax(0.11);
  it->set_ay(0.12);
  it->set_az(0.13);
  it->set_gx(0.14);
  it->set_gy(0.15);
  it->set_gz(0.16);

  EXPECT_TRUE(WriteImuFile(filename, imu));

  proto::ImuMsgList imu_out;
  EXPECT_TRUE(ReadImuFile(filename, imu_out));

  EXPECT_EQ(imu.imu_msgs_size(), imu_out.imu_msgs_size());
  for (int i = 0; i < imu.imu_msgs_size(); i++) {
    EXPECT_EQ(imu.imu_msgs(i).timestamp(), imu_out.imu_msgs(i).timestamp());
    EXPECT_EQ(imu.imu_msgs(i).ax(), imu_out.imu_msgs(i).ax());
    EXPECT_EQ(imu.imu_msgs(i).ay(), imu_out.imu_msgs(i).ay());
    EXPECT_EQ(imu.imu_msgs(i).az(), imu_out.imu_msgs(i).az());
    EXPECT_EQ(imu.imu_msgs(i).gx(), imu_out.imu_msgs(i).gx());
    EXPECT_EQ(imu.imu_msgs(i).gy(), imu_out.imu_msgs(i).gy());
    EXPECT_EQ(imu.imu_msgs(i).gz(), imu_out.imu_msgs(i).gz());
  }
}

TEST(SensorIO, ReadWriteEncoderFile) {
  std::string filename = "/tmp/motor.dat";

  proto::EncoderMsgList motor;
  auto it = motor.add_encoder_msgs();
  it->set_timestamp(1000);
  it->set_rx(0.1);
  it->set_ry(0.2);
  it->set_rz(0.3);
  it->set_rw(0.4);

  EXPECT_TRUE(WriteEncoderFile(filename, motor));

  proto::EncoderMsgList motor_out;
  EXPECT_TRUE(ReadEncoderFile(filename, motor_out));

  EXPECT_EQ(motor.encoder_msgs_size(), motor_out.encoder_msgs_size());
  for (int i = 0; i < motor.encoder_msgs_size(); i++) {
    EXPECT_EQ(motor.encoder_msgs(i).timestamp(),
              motor_out.encoder_msgs(i).timestamp());
    EXPECT_EQ(motor.encoder_msgs(i).rx(), motor_out.encoder_msgs(i).rx());
    EXPECT_EQ(motor.encoder_msgs(i).ry(), motor_out.encoder_msgs(i).ry());
    EXPECT_EQ(motor.encoder_msgs(i).rz(), motor_out.encoder_msgs(i).rz());
    EXPECT_EQ(motor.encoder_msgs(i).rw(), motor_out.encoder_msgs(i).rw());
  }
}

TEST(SensorIO, ReadWriteLidarFile) {
  std::string filename = "/tmp/lidar.dat";

  std::vector<ConstPtr<proto::LidarMsg>> lidar_msgs;
  for (int i = 0; i < 10; ++i) {
    Ptr<proto::LidarMsg> lidar_msg{new proto::LidarMsg()};
    for (int j = 0; j < 32000; ++j) {
      lidar_msg->add_points();
      lidar_msg->mutable_points(j)->set_x(j + 0.1);
      lidar_msg->mutable_points(j)->set_y(j + 0.2);
      lidar_msg->mutable_points(j)->set_z(j + 0.3);
      lidar_msg->mutable_points(j)->set_intensity(10);
      lidar_msg->mutable_points(j)->set_timestamp(1000);
    }
    lidar_msgs.push_back(lidar_msg);
  }

  EXPECT_TRUE(WriteLidarFile(filename, lidar_msgs));

  std::vector<ConstPtr<proto::LidarMsg>> lidar_msgs_out;
  EXPECT_TRUE(ReadLidarFile(
      filename, [&lidar_msgs_out](const ConstPtr<proto::LidarMsg> &lidar_msg) {
        lidar_msgs_out.push_back(lidar_msg);
      }));

  EXPECT_EQ(lidar_msgs.size(), lidar_msgs_out.size());
  for (int i = 0; i < lidar_msgs.size(); i++) {
    EXPECT_EQ(lidar_msgs[i]->points_size(), lidar_msgs_out[i]->points_size())
        << i;
    for (int j = 0; j < lidar_msgs[i]->points_size(); j++) {
      EXPECT_EQ(lidar_msgs[i]->points(j).x(), lidar_msgs_out[i]->points(j).x());
      EXPECT_EQ(lidar_msgs[i]->points(j).y(), lidar_msgs_out[i]->points(j).y());
      EXPECT_EQ(lidar_msgs[i]->points(j).z(), lidar_msgs_out[i]->points(j).z());
      EXPECT_EQ(lidar_msgs[i]->points(j).intensity(),
                lidar_msgs_out[i]->points(j).intensity());
      EXPECT_EQ(lidar_msgs[i]->points(j).timestamp(),
                lidar_msgs_out[i]->points(j).timestamp());
    }
  }
}
