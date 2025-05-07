#pragma once

#include "common/types.h"
#include "proto/sensors.pb.h"

void Optimize(std::vector<TimestampedPointCloud> &submaps,
              const proto::PgoConfig &config);
