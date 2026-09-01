#pragma once

#include <array>
#include <atomic>
#include <cstdint>
#include <memory>
#include <string>

#include <ulog_cpp/simple_writer.hpp>

#include "lio_msgs.h"

namespace middleware {

// Writes the same lio.ulg schema as xslam_core_ros so existing ULog
// inspection tools can consume Phoenix results without conversion.
class ULogStorage {
 public:
  ULogStorage() = default;
  ~ULogStorage();

  ULogStorage(const ULogStorage&) = delete;
  ULogStorage& operator=(const ULogStorage&) = delete;

  bool IsRunning() const { return initialized_.load(); }

  void StartLog(const std::string& log_filename);
  void StopLog();

  void HandleFullState(double timestamp, const lixel::FullStateMsg& msg);
  void HandleIeskfAttribute(double timestamp,
                            const lixel::AttributeIESKF& msg);
  void HandleIeskfStatePredict(int sweep_id,
                               const lixel::StatePredict& msgs);
  void HandleIeskfAttributePredict(int sweep_id,
                                   const lixel::AttributePredict& msgs);
  void HandleDebugMsgs(const lixel::DebugMsgs& msgs);

 private:
  enum ULoggerId {
    kFullState = 0,
    kIeskfAttribute,
    kIeskfStatePredict,
    kIeskfAttributePredict,
    kDebugMsgs,
    kLoggerCount
  };

  std::atomic<bool> initialized_{false};
  std::shared_ptr<ulog_cpp::SimpleWriter> writer_;
  std::array<uint16_t, kLoggerCount> message_ids_{};
};

}  // namespace middleware
