#pragma once

#include <functional>
#include <queue>
#include <thread>
#include <vector>
#include "common/common_lib.h"
#include "common/error_code.h"
#include "ieskf/ieskf.h"
#include "lio_msgs.h"
#include "method/io_utils.h"
#include "method/lio_init.h"
#include "method/preprocess.h"
#include "method/undistort.h"
#include "method/uniform_sampling.h"
#include "parameters.h"
#include "sensor_fusion/lidar_fusion.h"
#include "xmap.h"
#include "xmap_util.h"

namespace lixel
{

enum class ProcessState
{
  IDLE = 0,     // 未开始
  START = 1,    // 收到开始指令
  RUNNING = 2,  // 有数据
  STOP = 3      // 结束
};

constexpr int FORGET_FRAME_NUM = 10;

class LioCore
{
 public:
  EIGEN_MAKE_ALIGNED_OPERATOR_NEW

  typedef std::function<void(const LioResultMsg::Ptr &)> LioResultCallback;

  LioCore(LioParameters &lio_param);

  ~LioCore();

  void setParameters(LioParameters &lio_param);

  template <typename T>
  bool addSensorData(const T &msg);

  void setLioResultCallback(const LioResultCallback &callback_func);

  bool containsEnoughDataForSyncPackages();

  double getInputDataCommonDuration();

  void start();

  void stop();
  /*******************lio output end*******************/

 private:
  /*******************main process start*******************/
  void init();

  bool syncPackages(MeaureGroup &mg);

  void mappingLoop();

  void publisherLoop();
  /*******************main process end*******************/

 private:
  IOUtils io_utils_;
  LioParameters lio_param_;
  PreProcess preprocess_;
  std::shared_ptr<Initialization> initialization_;
  std::atomic<ProcessState> process_state_ = ProcessState::IDLE;
  std::atomic<bool> should_exit_ = false;
  std::shared_ptr<xmap::Xmap> xmap_ = nullptr;
  IESKF::Ptr ieskf_ptr_ = nullptr;
  LiDARFusion lidar_fusion_;
  // UniformSampling<PointXYZINormal> uniform_sampling_surf_;
  UniformSampling<PointXYZINormal> uniform_sampling_surf_;
  UniformSampling<PointXYZINormal> uniform_sampling_map_;

  std::thread mapping_thread_;
  std::thread publish_thread_;
  LioResultCallback lio_result_callback_;

  PointCloudXYZINormal::Ptr write_buf_ = nullptr;
};

}  // namespace lixel