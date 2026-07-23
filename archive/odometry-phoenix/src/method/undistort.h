#pragma once
#include <deque>
#include <vector>
#include "common/common_struct.h"
#include "common/error_code.h"
#include "lio_msgs.h"

namespace lixel
{

ErrorCodeType
undistort(const lixel::PointCloud &raw_data, StatePredict state, lixel::PointCloud &output);

}  // namespace lixel
