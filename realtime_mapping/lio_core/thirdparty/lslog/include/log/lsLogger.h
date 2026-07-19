#pragma once

#include <spdlog/spdlog.h>
#include <sstream>
#include "spdlog/sinks/stdout_color_sinks.h" // 对于带颜色的输出

#include "export.h"

XGRIDS_BEGIN_NAMESPACE

using xgridsLogLevel = spdlog::level::level_enum;

class  lsLogger {
 public:
  static void init(xgridsLogLevel log, const std::string &path, bool bCryptographic = false);
  
  static void flush() { logger->flush(); }


protected:
  template <class T>
  friend void debug(T &&);

  template <class T>
  friend void info(T &&);

  template <class T>
  friend void warn(T &&);

  template <class T>
  friend void error(T &&);

  template <class T>
  friend void console(T &&);

  template <typename... Args>
  friend void debug(Args &&...);

  template <typename... Args>
  friend void info(Args &&...);

  template <typename... Args>
  friend void warn(Args &&...);

  template <typename... Args>
  friend void error(Args &&...);

  template <typename... Args>
  friend void console(Args &&...);

  static spdlog::logger *logger;
};

class  hsLogEventWrap {
 public:
  enum class LogLevel {

    LSLOG_DEBUG = xgridsLogLevel::debug,
    LSLOG_INFO = xgridsLogLevel::info,
    LSLOG_WARNING = xgridsLogLevel::warn,
    LSLOG_ERROR = xgridsLogLevel::err,
    // CONSOLE = xgridsLogLevel::critical
  };

 public:
  hsLogEventWrap(LogLevel level) : level_(level) {}

  ~hsLogEventWrap() /*noexcept*/;
  hsLogEventWrap(const hsLogEventWrap&) = delete;
  hsLogEventWrap( hsLogEventWrap&&) = delete;

  std::stringstream &getSS() { return ss; }

 private:
  std::stringstream ss;
  LogLevel level_;
};

template <class T>
void debug(T &&msg) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->debug(std::forward<T>(msg));
    return;
  }
  lsLogger::logger->debug(std::forward<T>(msg));
}

template <class T>
void info(T &&msg) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->info(std::forward<T>(msg));
    return;
  }
  lsLogger::logger->info(std::forward<T>(msg));
}

template <class T>
void warn(T &&msg) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->warn(std::forward<T>(msg));
    return;
  }
  lsLogger::logger->warn(std::forward<T>(msg));
}

template <class T>
void error(T &&msg) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->error(std::forward<T>(msg));
    return;
  }
  lsLogger::logger->error(std::forward<T>(msg));
}

template <class T>
void console(T &&msg) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->critical(std::forward<T>(msg));
    return;
  }
  lsLogger::logger->critical(std::forward<T>(msg));
}

template <typename... Args>
void debug(Args &&...args) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->debug(std::forward<Args>(args)...);
    return;
  }
  lsLogger::logger->debug(std::forward<Args>(args)...);
}

template <typename... Args>
void info(Args &&...args) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->info(std::forward<Args>(args)...);
    return;
  }
  lsLogger::logger->info(std::forward<Args>(args)...);
}

template <typename... Args>
void warn(Args &&...args) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->warn(std::forward<Args>(args)...);
    return;
  }
  lsLogger::logger->warn(std::forward<Args>(args)...);
}

template <typename... Args>
void error(Args &&...args) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->error(std::forward<Args>(args)...);
    return;
  }
  lsLogger::logger->error(std::forward<Args>(args)...);
}

template <typename... Args>
void console(Args &&...args) {
  if(lsLogger::logger == nullptr) {
    spdlog::stderr_color_mt("stderr_logger")->critical(std::forward<Args>(args)...);
    return;
  }
  lsLogger::logger->critical(std::forward<Args>(args)...);
}
XGRIDS_END_NAMESPACE

#define lslog(level) xgrids_lio::hsLogEventWrap(xgrids_lio::hsLogEventWrap::LogLevel::level).getSS()
// #include <glog/logging.h>
