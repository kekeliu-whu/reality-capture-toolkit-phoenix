#include "io_utils.h"
#include "log/lsLogger.h"

namespace
{

template <typename T, typename TOut>
void collectImuOrMotorData(const std::deque<T> &data_queue, double start_time, double end_time, TOut &out_queue)
{
  int idx_beg = static_cast<int>(std::lower_bound(
                    data_queue.begin(),
                    data_queue.end(),
                    start_time,
                    [](const T &a, const double &b) { return a->timestamp < b; }) -
                data_queue.begin()) - 1;
  int idx_end = static_cast<int>(std::lower_bound(
                    data_queue.begin(),
                    data_queue.end(),
                    end_time,
                    [](const T &a, const double &b) { return a->timestamp < b; }) -
                data_queue.begin());
  // DCHECK_GE(idx_beg, 0);
  // DCHECK_LT(idx_beg, data_queue.size());
  // DCHECK_GE(idx_end, 0);
  // DCHECK_LT(idx_end, data_queue.size());
  // DCHECK_GE(idx_end, idx_beg);

  if (data_queue.empty()) return;
  idx_beg = std::max(idx_beg, 0);
  idx_end = std::min(idx_end, static_cast<int>(data_queue.size()) - 1);
  for (int i = idx_beg; i <= idx_end; ++i)
  {
    out_queue.push_back(*data_queue[i]);
  }
}

}  // namespace

namespace lixel
{

lixel::IOUtils::IOUtils(bool motor_enabled, bool sweep_cut_auto)
    : motor_enabled_(motor_enabled), sweep_cut_auto_(sweep_cut_auto)
{
}

lixel::IOUtils::~IOUtils()
{
}

void lixel::IOUtils::addLidar(const PointCloudMsg::Ptr &msg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  if (!msg->points)
  {
    lslog(LSLOG_ERROR) << "msg->points is NULL";
    return;
  }
  // todo add comments here
  double original_lidar_endtime = msg->points->back().timestamp;
  std::sort(
      msg->points->points.begin(),
      msg->points->points.end(),
      [](const auto &a, const auto &b) { return a.timestamp < b.timestamp; });
  int accepted_point_count = 0;
  for (int i = 0; i < msg->points->size(); ++i)
  {
    auto point = msg->points->points[i];
    if (point.timestamp > original_lidar_endtime)
    {
      continue;
    }
    if (!input_data_buff_.lidar_points_queue.empty())
    {
      // DCHECK_GE(point.timestamp, input_data_buff_.lidar_points_queue.back().timestamp)
      // << " lidar point timestamp is in wrong order " << std::fixed << std::setprecision(6)
      // << input_data_buff_.lidar_points_queue.back().timestamp << " " << point.timestamp;
    }
    input_data_buff_.lidar_points_queue.push_back(point);
    ++accepted_point_count;
  }
  input_data_buff_.lidar_frame_sizes_queue.push_back(accepted_point_count);
  input_data_buff_.lidar_frameids_queue.push_back(msg->frame_id);
}

void lixel::IOUtils::addImu(const ImuMsg::Ptr &msg)
{
  // TODO: verify the imu delay
  msg->timestamp -= 0.0025;
  static double first_timestamp = msg->timestamp;
  msg->timestamp += 0.003 - 5.333333333333334e-06 * (msg->timestamp - first_timestamp);
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  input_data_buff_.imu_queue.push_back(msg);
}

void lixel::IOUtils::addMotor(const MotorMsg::Ptr &msg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  input_data_buff_.motor_queue.push_back(msg);
}

void lixel::IOUtils::addGnss(const GnssMsg::Ptr &msg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  input_data_buff_.gnss_queue.push_back(msg);
}

void lixel::IOUtils::addImage(const ImageMsg::Ptr &msg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  input_data_buff_.image_queue.push_back(msg);
}

void lixel::IOUtils::addOdometry(const OdometryMsg::Ptr &msg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  input_data_buff_.prior_odom_queue.push_back(msg);
}

bool lixel::IOUtils::containsEnoughDataForSyncPackages(double sweep_duration)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  // if data queue is empty, return false
  if (input_data_buff_.lidar_points_queue.empty() || input_data_buff_.imu_queue.empty() ||
      (motor_enabled_ && input_data_buff_.motor_queue.empty()))
  {
    return false;
  }

  // if data queue duration is not enough, return false
  static bool trace_once = true;
  if (trace_once)
  {
    std::fprintf(stderr, "sync check queues lidar=%zu frames=%zu imu=%zu\n",
                 input_data_buff_.lidar_points_queue.size(), input_data_buff_.lidar_frame_sizes_queue.size(),
                 input_data_buff_.imu_queue.size());
    std::fflush(stderr);
  }
  double common_begin_time = getCommonBeginTime();
  double common_end_time = getCommonEndTime();
  if (trace_once)
  {
    std::fprintf(stderr, "sync check range %.9f %.9f\n", common_begin_time, common_end_time);
    std::fflush(stderr);
    trace_once = false;
  }
  if (common_end_time - common_begin_time <= sweep_duration + MIN_COMMON_QUEUE_DURATION)
  {
    return false;
  }

  return true;
}

bool lixel::IOUtils::syncPackagesImpl(double sweep_duration, MeaureGroup &mg)
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};

  if (this->sweep_cut_auto_)
  {
    if (!syncPackagesImplCutByDuration(sweep_duration, mg))
    {
      return false;
    }
  }
  else
  {
    if (!syncPackagesImplCutByOriginalSize(mg))
    {
      return false;
    }
  }

  input_data_buff_.lidar_points_queue.shrink_to_fit();
  input_data_buff_.imu_queue.shrink_to_fit();
  input_data_buff_.motor_queue.shrink_to_fit();

  return true;
}

void IOUtils::printSyncPackagesResult(MeaureGroup &mg)
{
  if (motor_enabled_)
  {
    lslog(LSLOG_INFO) << "syncPackage done: sweep_" << mg.sweep_id << " lidar_" << std::fixed << std::setprecision(6)
                      << mg.lidar_points->points.back().timestamp - mg.lidar_points->points.front().timestamp << "["
                      << mg.lidar_points->points.front().timestamp << "," << mg.lidar_points->points.back().timestamp
                      << "] imu_" << mg.imu_vec.back().timestamp - mg.imu_vec.front().timestamp << "["
                      << mg.imu_vec.front().timestamp << "," << mg.imu_vec.back().timestamp << "] motor_"
                      << mg.motor_vec.back().timestamp - mg.motor_vec.front().timestamp << "["
                      << mg.motor_vec.front().timestamp << "," << mg.motor_vec.back().timestamp << "]";
  }
  else
  {
    lslog(LSLOG_INFO) << "syncPackage done: sweep_" << mg.sweep_id << " lidar_" << std::fixed << std::setprecision(6)
                      << mg.lidar_points->points.back().timestamp - mg.lidar_points->points.front().timestamp << "["
                      << mg.lidar_points->points.front().timestamp << "," << mg.lidar_points->points.back().timestamp
                      << "] imu_" << mg.imu_vec.back().timestamp - mg.imu_vec.front().timestamp << "["
                      << mg.imu_vec.front().timestamp << "," << mg.imu_vec.back().timestamp << "]";
  }
}

void IOUtils::syncPackagesImplHandleImu(MeaureGroup &mg)
{
  collectImuOrMotorData(input_data_buff_.imu_queue, mg.pcl_start_time, mg.pcl_end_time, mg.imu_vec);

  // trim imu queue and motor queue to mg.pcl_start_time-1
  while (!input_data_buff_.imu_queue.empty())
  {
    if (input_data_buff_.imu_queue.front()->timestamp > mg.pcl_start_time - PADDING_DURATION_FOR_IMU_MOTOR_BUFFER)
    {
      break;
    }
    input_data_buff_.imu_queue.pop_front();
  }

  // DCHECK_LT(mg.imu_vec[0].timestamp, mg.pcl_start_time);
  // DCHECK_LE(mg.pcl_start_time, mg.imu_vec[1].timestamp);
  // DCHECK_GE(mg.imu_vec.size(), 2);
  // DCHECK_LT(mg.imu_vec[mg.imu_vec.size() - 2].timestamp, mg.pcl_end_time);
  // DCHECK_LE(mg.pcl_end_time, mg.imu_vec[mg.imu_vec.size() - 1].timestamp);
}

bool IOUtils::syncPackagesImplCutByDuration(double sweep_duration, MeaureGroup &mg)
{
  if (!containsEnoughDataForSyncPackages(sweep_duration))
  {
    return false;
  }

  // trim lidar points to common begin time
  static bool first_frame = true;
  static double last_pcl_end_time;
  double common_begin_time = getCommonBeginTime();
  if (first_frame)
  {
    while (!input_data_buff_.lidar_points_queue.empty())
    {
      if (input_data_buff_.lidar_points_queue.front().timestamp > common_begin_time)
      {
        break;
      }
      input_data_buff_.lidar_points_queue.pop_front();
    }
  }

  // collect data to MeaureGroup
  mg.lidar_points.reset(new PointCloud);
  mg.imu_vec.clear();
  mg.motor_vec.clear();

  while (!input_data_buff_.lidar_points_queue.empty())
  {
    if (input_data_buff_.lidar_points_queue.front().timestamp > common_begin_time + sweep_duration)
    {
      break;
    }
    mg.lidar_points->push_back(input_data_buff_.lidar_points_queue.front());
    input_data_buff_.lidar_points_queue.pop_front();
  }

  mg.pcl_start_time = first_frame ? mg.lidar_points->front().timestamp : last_pcl_end_time;
  mg.pcl_end_time = mg.lidar_points->back().timestamp;
  first_frame = false;
  last_pcl_end_time = mg.pcl_end_time;

  mg.sweep_id = sweep_id++;

  syncPackagesImplHandleImu(mg);
  syncPackagesImplHandleMotor(mg);
  printSyncPackagesResult(mg);

  return true;
}

bool IOUtils::syncPackagesImplCutByOriginalSize(MeaureGroup &mg)
{
  if (input_data_buff_.lidar_points_queue.empty() || input_data_buff_.imu_queue.empty() ||
      (motor_enabled_ && input_data_buff_.motor_queue.empty()) ||
      this->input_data_buff_.lidar_frame_sizes_queue.empty())
  {
    return false;
  }

  // if data queue duration is not enough, return false
  double common_begin_time = getCommonBeginTime();
  double common_end_time = getCommonEndTime();
  if (common_end_time - common_begin_time <= MIN_COMMON_QUEUE_DURATION)
  {
    return false;
  }

  // trim lidar points to common begin time
  static bool first_frame = true;
  static double last_pcl_end_time;
  if (input_data_buff_.lidar_points_queue.front().timestamp < common_begin_time)
  {
    // remove a frame from lidar points
    for (int i = 0; i < input_data_buff_.lidar_frame_sizes_queue.front(); ++i)
    {
      input_data_buff_.lidar_points_queue.pop_front();
    }
    input_data_buff_.lidar_frame_sizes_queue.pop_front();
    return false;
  }

  // collect data to MeaureGroup
  mg.lidar_points.reset(new PointCloud);
  mg.imu_vec.clear();
  mg.motor_vec.clear();

  lslog(LSLOG_INFO) << "offline sync: points=" << input_data_buff_.lidar_frame_sizes_queue.front()
                    << " lidar_queue=" << input_data_buff_.lidar_points_queue.size()
                    << " imu_queue=" << input_data_buff_.imu_queue.size()
                    << " common=[" << std::fixed << common_begin_time << "," << common_end_time << "]";

  for (int i = 0; i < input_data_buff_.lidar_frame_sizes_queue.front(); ++i)
  {
    mg.lidar_points->push_back(input_data_buff_.lidar_points_queue.front());
    input_data_buff_.lidar_points_queue.pop_front();
  }
  input_data_buff_.lidar_frame_sizes_queue.pop_front();

  mg.pcl_start_time = first_frame ? mg.lidar_points->front().timestamp : last_pcl_end_time;
  mg.pcl_end_time = mg.lidar_points->back().timestamp;
  first_frame = false;
  last_pcl_end_time = mg.pcl_end_time;

  mg.sweep_id = input_data_buff_.lidar_frameids_queue.front();
  input_data_buff_.lidar_frameids_queue.pop_front();

  lslog(LSLOG_INFO) << "offline sync: lidar collected [" << std::fixed << mg.pcl_start_time << ","
                    << mg.pcl_end_time << "]";
  syncPackagesImplHandleImu(mg);
  lslog(LSLOG_INFO) << "offline sync: imu collected=" << mg.imu_vec.size();
  syncPackagesImplHandleMotor(mg);
  printSyncPackagesResult(mg);

  return true;
}

void lixel::IOUtils::addLioResult(const LioResultMsg::Ptr &lio_result)
{
  std::lock_guard<std::mutex> lg{mtx_output_data_buff_};
  output_data_buff_.lio_result_queue.push(lio_result);
}

lixel::LioResultMsg::Ptr lixel::IOUtils::getLioResult()
{
  std::lock_guard<std::mutex> lg{mtx_output_data_buff_};
  if (output_data_buff_.lio_result_queue.empty())
    return NULL;

  LioResultMsg::Ptr lio_result = output_data_buff_.lio_result_queue.front();
  output_data_buff_.lio_result_queue.pop();
  return lio_result;
}

double lixel::IOUtils::getInputDataCommonDuration()
{
  std::lock_guard<std::recursive_mutex> lg{mtx_input_data_buff_};
  if (input_data_buff_.imu_queue.empty() || input_data_buff_.lidar_points_queue.empty() ||
      (motor_enabled_ && input_data_buff_.motor_queue.empty()))
  {
    return 0.0;
  }
  double common_begin_time = getCommonBeginTime();
  double common_end_time = getCommonEndTime();
  return common_end_time - common_begin_time;
}

double lixel::IOUtils::getCommonBeginTime()
{
  std::vector<double> arr;
  arr.push_back(input_data_buff_.imu_queue.front()->timestamp);
  if (motor_enabled_)
  {
    arr.push_back(input_data_buff_.motor_queue.front()->timestamp);
  }
  arr.push_back(input_data_buff_.lidar_points_queue.front().timestamp);
  return *std::max_element(std::begin(arr), std::end(arr));
}

double lixel::IOUtils::getCommonEndTime()
{
  std::vector<double> arr;
  arr.push_back(input_data_buff_.imu_queue.back()->timestamp);
  if (motor_enabled_)
  {
    arr.push_back(input_data_buff_.motor_queue.back()->timestamp);
  }
  arr.push_back(input_data_buff_.lidar_points_queue.back().timestamp);
  return *std::min_element(std::begin(arr), std::end(arr));
}

void IOUtils::syncPackagesImplHandleMotor(MeaureGroup &mg)
{
  if (!motor_enabled_)
  {
    return;
  }

  collectImuOrMotorData(input_data_buff_.motor_queue, mg.pcl_start_time, mg.pcl_end_time, mg.motor_vec);

  while (!input_data_buff_.motor_queue.empty())
  {
    if (input_data_buff_.motor_queue.front()->timestamp > mg.pcl_start_time - PADDING_DURATION_FOR_IMU_MOTOR_BUFFER)
    {
      break;
    }
    input_data_buff_.motor_queue.pop_front();
  }

  // DCHECK_LT(mg.motor_vec[0].timestamp, mg.pcl_start_time);
  // DCHECK_LE(mg.pcl_start_time, mg.motor_vec[1].timestamp);
  // DCHECK_GE(mg.motor_vec.size(), 2);
  // DCHECK_LT(mg.motor_vec[mg.motor_vec.size() - 2].timestamp, mg.pcl_end_time);
  // DCHECK_LE(mg.pcl_end_time, mg.motor_vec[mg.motor_vec.size() - 1].timestamp);
}

}  // namespace lixel
