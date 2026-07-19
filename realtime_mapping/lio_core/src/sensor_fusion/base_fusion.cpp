//
// Created by youyuan on 24-2-18.
//
#include "sensor_fusion/base_fusion.h"

namespace lixel
{

void BaseFusion::setCurState(KFState& states_group)
{
  states_group_ptr_ = &states_group;
}

}  // namespace lixel