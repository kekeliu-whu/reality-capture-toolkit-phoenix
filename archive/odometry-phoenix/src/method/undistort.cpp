#include "undistort.h"
#include <glog/logging.h>

namespace
{

std::vector<double> getImuTime(const lixel::StatePredict &states)
{
  std::vector<double> time;
  time.reserve(states.size());
  for (auto state : states)
  {
    time.push_back(state.timestamp);
  }
  return time;
}

std::vector<double> getCloudTime(const lixel::PointCloud &cloud);

/**
 * @brief get the data's closer right bound in edges, the bound size is same to data
 *
 * @param[in] data the data need to find right bound
 * @param[in] edges the edges is closure for data
 * @return std::vector<int> the data's closer right bound in edges
 */
// TODO: current method is log2(N), possible to improve
std::vector<int> discretize(const std::vector<double> &data, const std::vector<double> &edges)
{
  if ((data.front() < edges.front()) || (data.back() > edges.back()))
  {
    std::vector<int> discretized(0);
    return discretized;
  }
  std::vector<int> discretized(data.size());
  for (size_t i = 0; i < data.size(); ++i)
  {
    auto it = std::lower_bound(edges.begin(), edges.end(), data[i]);
    discretized[i] = std::distance(edges.begin(), it);
  }
  return discretized;
}

std::vector<double> getCloudTime(const lixel::PointCloud &cloud)
{
  std::vector<double> time;
  time.reserve(cloud.height);
  for (auto point : cloud)
  {
    time.push_back(point.timestamp);
  }
  return time;
}

}  // namespace

namespace lixel
{

ErrorCodeType undistort(const lixel::PointCloud &raw_cloud, StatePredict state, lixel::PointCloud &output)
{
  auto cloud_time = getCloudTime(raw_cloud);
  if (cloud_time.empty())
  {
    LOG(ERROR) << "cloud_time is empty";
    return Error::LioCore::Undistort::RAW_POINT_CLOUD_IS_EMPTY;
  }
  auto imu_time = getImuTime(state);
  if (imu_time.empty())
  {
    LOG(ERROR) << "imu_time is empty";
    return Error::LioCore::Undistort::PREDICT_STATE_IS_EMPTY;
  }
  auto edge_index = discretize(cloud_time, imu_time);
  if (edge_index.empty())
  {
    LOG(ERROR) << "edge_index is empty";
    return Error::LioCore::Undistort::INVALID_UNDISTORT_TIME_RANGE;
  }
  output = raw_cloud;
  lixel::QUATF quat_end_inv = lixel::QUATF::Identity();
  lixel::V3F pos_end = lixel::V3F::Zero();
  for (int i = output.size() - 1; i >= 0; --i)
  {
    auto xyz = output[i].getVector3fMap();
    auto cur_id = edge_index[i] - 1;
    auto next_id = edge_index[i];
    auto radio = static_cast<float>((output[i].timestamp - imu_time[cur_id]) / (imu_time[next_id] - imu_time[cur_id]));
    auto pos = state[cur_id].pos + (state[next_id].pos - state[cur_id].pos) * radio;
    auto quat = state[cur_id].quat.slerp(radio, state[next_id].quat);
    xyz = quat * xyz + pos;  // world frame xyz

    if (i == output.size() - 1)
    {
      quat_end_inv = quat.inverse();
      pos_end = pos;
    }
    xyz = quat_end_inv * (xyz - pos_end);  // body frame xyz
  }
  return Error::SUCCESS;
}

}  // namespace lixel