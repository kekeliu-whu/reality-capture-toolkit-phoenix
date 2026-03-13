#pragma once

#include "common/types.h"
#include "proto/sensors.pb.h"

class PgoRunner {
public:
  PgoRunner(proto::PgoConfig config) : config_(config),
                                       use_rtk_constraint_(false) {}

  void Run(const std::string &input_path, const std::string &output_path);

  /**
   * @brief Run optimization with GNSS/RTK data
   * 
   * @param input_path Path to project directory
   * @param output_path Path to output directory
   * @param use_gnss Whether to use GNSS data if available
   */
  void RunWithGnss(const std::string &input_path, 
                   const std::string &output_path,
                   bool use_gnss = true);

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
