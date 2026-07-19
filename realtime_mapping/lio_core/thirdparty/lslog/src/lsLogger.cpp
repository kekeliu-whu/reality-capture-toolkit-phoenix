#include "log/lsLogger.h"
// #include <QTextEdit>
#include <memory>
#include "cryptographic_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"

using namespace xgrids_lio;

spdlog::logger *lsLogger::logger = nullptr;

void lsLogger::init(xgridsLogLevel log, const std::string &path, bool bCryptographic)
{
  auto sink1 = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  spdlog::logger *logger_ptr = new spdlog::logger("", {sink1});

  sink1->set_level(log);
  if (bCryptographic)
  {
    auto sink2 = std::make_shared<spdlog::sinks::cryptographic_sink_mt>(path);
    sink2->set_level(log);
    logger_ptr->sinks().push_back(sink2);
  }
  else
  {
    auto sink2 = std::make_shared<spdlog::sinks::basic_file_sink_mt>(path);
    sink2->set_level(log);
    logger_ptr->sinks().push_back(sink2);
  }
  logger_ptr->flush();

  logger = logger_ptr;
}

hsLogEventWrap::~hsLogEventWrap() /*noexcept*/
{
  switch (level_)
  {
    case LogLevel::LSLOG_INFO:
      info(ss.str());
      break;
    case LogLevel::LSLOG_DEBUG:
      debug(ss.str());
      break;
    case LogLevel::LSLOG_WARNING:
      warn(ss.str());
      break;
    case LogLevel::LSLOG_ERROR:
      error(ss.str());
      break;
      //        case LogLevel::CONSOLE:
      //            error(ss.str());
      //            break;
    default:
      break;
  }
}
