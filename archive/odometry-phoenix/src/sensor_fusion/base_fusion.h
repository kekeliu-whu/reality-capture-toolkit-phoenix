//
// Created by youyuan on 24-2-18.
//

#pragma once

#include <Eigen/Core>
#include <Eigen/Dense>
#include "common/common_struct.h"
#include "ieskf/state_group.h"
namespace lixel
{

class BaseFusion
{
 public:
  void setCurState(KFState& states_group);
  virtual void calculateMeas(VecX& residual, SparseMat& H, double& R) = 0;

 protected:
  const KFState* states_group_ptr_ = nullptr;
};
}  // namespace lixel
