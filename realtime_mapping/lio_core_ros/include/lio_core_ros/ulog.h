#pragma once

#include <memory>
#include <string>

#include "ulog_cpp/simple_writer.hpp"
#include "lio_msgs.h"

namespace middleware
{
class ULogStorage
{
 public:
  ULogStorage();
  virtual ~ULogStorage();
  bool IsRuning()
  {
    return m_init_done.load();
  }

  void stopLog();

  void startLog(std::string path);

  void HandleFullState(const lixel::FullStateMsg &msg);

  void HandleIeskfAttribute(const lixel::AttributeIESKF &msg);

  void HandleIeskfStatePredict(int sweep_id, const lixel::StatePredict &msgs);

  void HandleIeskfAttributePredict(int sweep_id, const lixel::AttributePredict &msgs);

  void HandleDebugMsgs(const lixel::DebugMsgs &msgs);

 private:
  enum ULoggerID
  {
    ULG_FullState = 0,
    ULG_IeskfAttribute,
    ULG_IeskfStatePredict,
    ULG_IeskfAttributePredict,
    ULG_DebugMsgs,
    ULG_MAX
  };

 private:
  std::string m_logPath;
  std::atomic<bool> m_init_done{false};
  std::shared_ptr<ulog_cpp::SimpleWriter> m_ulogger;
  int m_ulogIds[ULG_MAX];
};
}  // namespace middleware
