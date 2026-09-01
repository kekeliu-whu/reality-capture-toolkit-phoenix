#include "lio.h"
#include <pcl/common/common.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
// #include <liblas/liblas.hpp>
namespace
{
void setLioStatus(const lixel::KFState::ConstPtr &state, lixel::LioResultMsg::Ptr &lio_result)
{
  lio_result->full_state.q = lixel::QUATD(state->sw_rot_.back().cast<double>());
  lio_result->full_state.p = state->sw_pos_.back().cast<double>();
  lio_result->full_state.v = state->vel_.cast<double>();
  lio_result->full_state.ba = state->acc_bias_.cast<double>();
  lio_result->full_state.bg = state->gyo_bias_.cast<double>();
#if GRAVITY_CALIBRATION
  lio_result->full_state.gravity = lixel::V3D(state->gravity);
#else
  lio_result->full_state.gravity = lixel::DEFAULT_GRIVITY_VEC.cast<double>();
#endif
  lio_result->full_state.timestamp = state->sw_timestamp.back();
}
}  // namespace

namespace lixel
{
LioCore::LioCore(LioParameters &lio_param)
    : io_utils_(lio_param.extrinsic_param.motor_param.enabled,
                lio_param.preprocess_param.sweep_cut_auto,
                lio_param.sensor_param.imu_param.lidar_to_imu_time_offset_seconds,
                lio_param.sensor_param.imu_param.clock_drift_ppm),
      lidar_fusion_(lio_param.kf_param)
{
  init();
  setParameters(lio_param);
  xmap_ = std::make_shared<xmap::Xmap>(lio_param.map_param.config_path);
  ieskf_ptr_ = std::make_shared<IESKF>(lio_param_.kf_param);
  write_buf_.reset(new PointCloudXYZINormal);
  initialization_ = std::make_shared<Initialization>(
      lio_param_.init_param, lio_param_.kf_param.window_size);
  initialization_->setXmap(xmap_);
  initialization_->setIESKF(ieskf_ptr_);
  lidar_fusion_.setXmap(xmap_);
  LOG(INFO) << "init success!";
}

LioCore::~LioCore()
{
  LOG(INFO) << "lio_core exiting ...";
  stop();
  LOG(INFO) << "lio_core exit success";
}

void LioCore::init()
{
  mapping_thread_ = std::thread(&LioCore::mappingLoop, this);
  publish_thread_ = std::thread(&LioCore::publisherLoop, this);
}

void LioCore::setParameters(LioParameters &lio_param)
{
  lio_param_ = lio_param;
}

template <>
bool LioCore::addSensorData(const ImuMsg::Ptr &msg)
{
  io_utils_.addImu(msg);
  return true;
}

template <>
bool LioCore::addSensorData(const MotorMsg::Ptr &msg)
{
  io_utils_.addMotor(msg);
  return true;
}

template <>
bool LioCore::addSensorData(const PointCloudMsg::Ptr &msg)
{
  io_utils_.addLidar(msg);
  return true;
}

template <>
bool LioCore::addSensorData(const GnssMsg::Ptr &msg)
{
  io_utils_.addGnss(msg);
  return true;
}

void LioCore::setLioResultCallback(const LioResultCallback &callback_func)
{
  lio_result_callback_ = callback_func;
}

void LioCore::start()
{
  process_state_ = ProcessState::START;
  uniform_sampling_map_.setRandomSeed(0);
}

void LioCore::stop()
{
  process_state_ = ProcessState::STOP;  // flag stop and wait for stop in the destructor
  should_exit_ = true;
  LOG(INFO) << "lio_core exiting from stop() ...";
  if (mapping_thread_.joinable())
  {
    mapping_thread_.join();
  }
  if (publish_thread_.joinable())
  {
    publish_thread_.join();
  }
  LOG(INFO) << "lio_core exit success from stop()";

  pcl::io::savePLYFileBinary("/home/sandyyu/Desktop/test/cloud.ply", *write_buf_);
  write_buf_->clear();
}

bool LioCore::syncPackages(MeaureGroup &mg)
{
  if (ProcessState::IDLE == process_state_ || ProcessState::STOP == process_state_)
    return false;

  return io_utils_.syncPackagesImpl(this->lio_param_.preprocess_param.sweep_duration, mg);
}

bool LioCore::containsEnoughDataForSyncPackages()
{
  return io_utils_.containsEnoughDataForSyncPackages(this->lio_param_.preprocess_param.sweep_duration);
}

double lixel::LioCore::getInputDataCommonDuration()
{
  return io_utils_.getInputDataCommonDuration();
}

void LioCore::mappingLoop()
{
  std::chrono::milliseconds dura(5);
  MeaureGroup measurement;

  while (ProcessState::STOP != process_state_ && !should_exit_)
  {
    ErrorCodeType error_code;
    TicToc syncPackagesT;
    if (!syncPackages(measurement))
    {
      // TODO: remove this write io function
      std::this_thread::sleep_for(dura);
      continue;
    }
    LOG(INFO) << "syncPackagesT:" << syncPackagesT.Toc();

    // 0. check
    TicToc totalT;
    LOG(INFO) << std::fixed << "measurement.pcl_end_time:" << measurement.pcl_end_time;
    LOG(INFO) << std::fixed << "measurement.pcl_start_time:" << measurement.pcl_start_time;

    // 1. preprocess
    TicToc preprocessT;
    preprocess_.process(lio_param_, measurement);
    LOG(INFO) << "preprocessT:" << preprocessT.Toc();

    // 2. initialize
    if (!initialization_->initialize(measurement.imu_vec, measurement.lidar_points, measurement.pcl_end_time))
      continue;

    // 3. kf.predict
    // TODO:
    TicToc predictT;
    StatePredict state_predict;
    AttributePredict attr_predict;
    ieskf_ptr_->predict(
        measurement.imu_vec, measurement.pcl_end_time, measurement.sweep_id, state_predict, attr_predict);
    LOG(INFO) << "predictT:" << predictT.Toc();

    // 4. undistort
    TicToc undistortT;
    PointCloud::Ptr undistort_pcl(new PointCloud);
    error_code = undistort(*measurement.lidar_points, state_predict, *undistort_pcl);
    PointCloudXYZINormal::Ptr undistort_pcl_temp = transformPCL(undistort_pcl);
    LOG(INFO) << "undistortT:" << undistortT.Toc();

    // 5. downsample
    TicToc downsampleT;
    PointCloudXYZINormal::Ptr downsample_map_body(new PointCloudXYZINormal);
    PointCloudXYZINormal::Ptr downsample_surf_body(new PointCloudXYZINormal);
    uniform_sampling_map_.setRadius(lio_param_.downsample_param.base_downsample_dis);
    uniform_sampling_map_.setInputCloud(undistort_pcl_temp);
    uniform_sampling_map_.filter(*downsample_map_body);

    float adaptive_size = uniform_sampling_surf_.calculateRadius(downsample_map_body, lio_param_.downsample_param);
    uniform_sampling_surf_.setRadius(adaptive_size);
    uniform_sampling_surf_.setInputCloud(downsample_map_body);
    uniform_sampling_surf_.filter(*downsample_surf_body);

    LOG(INFO) << "sweep_id:" << measurement.sweep_id;
    LOG(INFO) << "originSize:" << measurement.lidar_points->size();
    LOG(INFO) << "undistortSize:" << undistort_pcl->size();
    LOG(INFO) << "downMapSize:" << downsample_map_body->size();
    LOG(INFO) << "downSurfSize:" << downsample_surf_body->size();
    LOG(INFO) << "downsampleT:" << downsampleT.Toc();

    // 6.7. construct H; kf.update
    TicToc updateT;
    downsample_surf_body->header.stamp = measurement.pcl_end_time * 1e6;
    undistort_pcl->header.stamp = measurement.pcl_end_time * 1e6;
    lidar_fusion_.setLidarMeas(downsample_surf_body, downsample_map_body, undistort_pcl);
    AttributeIterate attr_iter{};
    const bool update_success = ieskf_ptr_->update(lidar_fusion_, attr_iter);
    LOG(INFO) << "updateT:" << updateT.Toc();

    // 8.1 assign normal to point
    const std::unordered_map<size_t, Leaf> &index_leaf_map = uniform_sampling_surf_.getIndexMap();
    for (auto &pair : index_leaf_map)
    {
      const Leaf &value = pair.second;
      PointXYZINormal &point_search = downsample_surf_body->points[value.idx_in_output];
      std::vector<int> neighbor_index = value.indices_in_input;
      neighbor_index.push_back(value.idx_in_input);
      for (int index : neighbor_index)
      {
        downsample_map_body->points[index].normal_x = point_search.normal_x;
        downsample_map_body->points[index].normal_y = point_search.normal_y;
        downsample_map_body->points[index].normal_z = point_search.normal_z;
      }
    }
    size_t normal_assigned_count = 0;
    for (const PointXYZINormal &point : downsample_map_body->points)
    {
      if (point.getNormalVector3fMap().squaredNorm() > 1e-8f)
        ++normal_assigned_count;
    }
    const float normal_assigned_ratio = downsample_map_body->empty()
        ? 0.0f
        : static_cast<float>(normal_assigned_count) /
              static_cast<float>(downsample_map_body->size());
    AttributeIESKF frontend_attribute{};
    frontend_attribute.sweep_id = measurement.sweep_id;
    frontend_attribute.timestamp = measurement.pcl_end_time;
    frontend_attribute.update_success = update_success;
    frontend_attribute.downsample_dis = adaptive_size;
    frontend_attribute.state_predict = state_predict;
    frontend_attribute.attritube_predict = attr_predict;
    frontend_attribute.jacobi = lidar_fusion_.getAttributeJacobi();
    frontend_attribute.jacobi.normal_assigned_ratio = normal_assigned_ratio;
    frontend_attribute.iterate = attr_iter;

    // 8.2 update map
    TicToc mapT;
    PointCloudXYZINormal::Ptr update_pcl;
    bool need_update_map = lidar_fusion_.getUpdateFrame(update_pcl);
    PointCloudXYZINormal::Ptr downsample_map_world(new PointCloudXYZINormal);
    if (need_update_map && update_success)
    {
      transformToWorld(update_pcl, ieskf_ptr_->getStatesPtr(), downsample_map_world);
      downsample_map_world->header.stamp = measurement.pcl_end_time * 1e6;
      xmap_->mapIncremental(
          downsample_map_world,
          ieskf_ptr_->getStatesPtr()->sw_pos_.back().cast<xmap::FloatDataType>());
      LOG(INFO) << "mapT:" << mapT.Toc();
    }

    // 9. map forget
    if (measurement.sweep_id % FORGET_FRAME_NUM == 0)
    {
      xmap_->forget(measurement.pcl_end_time, ieskf_ptr_->getStatesPtr()->sw_pos_[0].cast<xmap::FloatDataType>());
    }

    // 10. publish result
    PointCloud::Ptr publish_pcl;
    bool need_publish = lidar_fusion_.getPublishFrame(publish_pcl);
    if (need_publish)
    {
      LioResultMsg::Ptr lio_result(new LioResultMsg());
      lio_result->attribute_ieskf = std::move(frontend_attribute);
      setLioStatus(ieskf_ptr_->getStatesPtr(), lio_result);
      lio_result->body_points = publish_pcl;
      io_utils_.addLioResult(lio_result);
    }
    if (process_state_ != ProcessState::STOP)
    {
      process_state_ = ProcessState::RUNNING;
    }
    LOG(INFO) << "totalT:" << totalT.Toc();
  }
  mapping_finished_ = true;
  LOG(INFO) << "exit from mappingLoop";
}

void LioCore::publisherLoop()
{
  std::chrono::milliseconds dura(10);

  while (1)  // todo control publisher loop
  {
    LioResultMsg::Ptr lio_result = io_utils_.getLioResult();
    if (lio_result_callback_ && lio_result)
    {
      lio_result_callback_(lio_result);
    }
    else
    {
      // exit only when lio_result queue is empty
      if (ProcessState::STOP == process_state_ && should_exit_ && mapping_finished_)
      {
        LOG(INFO) << "exit from publisherLoop";
        break;
      }
    }

    std::this_thread::sleep_for(dura);
  }
}

}  // namespace lixel
