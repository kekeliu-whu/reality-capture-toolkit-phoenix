#pragma once

#include "proto/sensors.pb.h"

class PgoRunner {
public:
  PgoRunner(proto::PgoConfig config) : config_(config) {}

  void Run(const std::string &input_path, const std::string &output_path);

private:
  proto::PgoConfig config_;
};
