#include "lio_core_ros/data_io.h"

DataIO::DataIO(const std::string& output_dir) : output_dir_(output_dir)
{
}

void DataIO::AddUndistortedLidarScan(const sensor_msgs::PointCloud2& scan)
{
  std::lock_guard<std::mutex> lg{mtx_};
  TryInit();

  xbc_writer_->write(TOPIC_PC_BIN, scan.header.stamp, scan);
}

void DataIO::AddLocalLioStates(const lixel_msgs::LioFullStates& lio_state)
{
  std::lock_guard<std::mutex> lg{mtx_};
  TryInit();

  xbc_writer_->write(TOPIC_FULL_STATES, lio_state.header.stamp, lio_state);
#ifdef __linux__
  poses_csv_ << std::fixed << std::setprecision(6) << lio_state.header.stamp << " " << lio_state.p.x << " "
             << lio_state.p.y << " " << lio_state.p.z << " " << lio_state.q.w << " " << lio_state.q.x << " "
             << lio_state.q.y << " " << lio_state.q.z << " " << lio_state.v.x << " " << lio_state.v.y << " "
             << lio_state.v.z << " " << lio_state.grav.x << " " << lio_state.grav.y << " " << lio_state.grav.z
             << std::endl;
  poses_csv_.flush();
#endif
}

void DataIO::Close()
{
  std::lock_guard<std::mutex> lg{mtx_};
  if (xbc_writer_)
  {
    xbc_writer_->close();
    xbc_writer_ = nullptr;
  }
}

DataIO::~DataIO()
{
  Close();
}

void DataIO::TryInit()
{
  if (xbc_writer_ == nullptr)
  {
    xbc_writer_.reset(new rosbag::Bag);
    // todo handle file open failed error
    xbc_writer_->open(output_dir_ + "/lio.xbc", rosbag::BagMode::Write);

#ifdef __linux__
    poses_csv_ = std::ofstream(output_dir_ + "/lio_poses.csv");
#endif
  }
}

void DataIO::AddCorrectedImu(const sensor_msgs::Imu& imu)
{
  std::lock_guard<std::mutex> lg{mtx_};
  TryInit();

  xbc_writer_->write(TOPIC_IMU_CORRECTED, imu.header.stamp, imu);
}
