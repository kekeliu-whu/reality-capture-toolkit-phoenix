#pragma once

#include <google/logging.h>

class EncryptedLogSink : public google::LogSink {
 public:
  void send(google::LogSeverity severity, const char* full_filename, const char* base_filename, int line, const struct ::tm* tm_time,
            const char* message, size_t message_len) override;
};
