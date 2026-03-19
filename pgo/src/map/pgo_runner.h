#pragma once

#include "common/types.h"
#include "proto/sensors.pb.h"

class PgoRunner {
public:
  PgoRunner(proto::PgoConfig config) : config_(config),
                                       use_rtk_constraint_(false),
                                       use_btc_constraint_(false) {}

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

  /**
   * @brief Enable/disable BTC (Binary Triangle Cluster) based loop closure
   * 
   * When enabled, uses the BTC library to detect loop closures based on
   * structural descriptors of point clouds. This enables detection of loops
   * between spatially close but temporally distant submaps.
   */
  void SetUseBtcConstraint(bool enable) {
    use_btc_constraint_ = enable;
  }

  bool GetUseBtcConstraint() const {
    return use_btc_constraint_;
  }

private:
  proto::PgoConfig config_;
  bool use_rtk_constraint_;  // Enable/disable RTK constraints
  bool use_btc_constraint_;  // Enable/disable BTC loop closure constraints
};
