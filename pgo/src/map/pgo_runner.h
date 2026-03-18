#pragma once

#include "common/types.h"
#include "proto/sensors.pb.h"

class PgoRunner {
public:
  PgoRunner(proto::PgoConfig config) : config_(config),
                                       use_rtk_constraint_(false) {}

  void Run(const std::string &input_path, const std::string &output_path);

  /**
   * @brief Enable/disable RTK constraint
   * 
   * When enabled, GNSS data will be used as pose constraints.
   * The first GNSS point's position is automatically used as the origin
   * for LocalENU coordinate system conversion.
   */
  void SetUseRtkConstraint(bool enable) {
    use_rtk_constraint_ = enable;
  }

  bool GetUseRtkConstraint() const {
    return use_rtk_constraint_;
  }

private:
  proto::PgoConfig config_;
  bool use_rtk_constraint_;  // Enable/disable RTK constraints
};
