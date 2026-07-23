//
// Created by youyuan on 24-2-18.
//

#pragma once

#define UPDATE_ROT false
#include "base_fusion.h"
namespace lixel
{

constexpr double STATIC_COV = 1e-6;
constexpr int STATIC_MEASUREMENT_DIM = 9;
class StaticFusion : public BaseFusion
{
 public:
  void calculateMeas(VecX& residual, SparseMat& H, double& R) override;
  void setInitRT(const Mat3& R, const Vec3& t);

 public:
  Mat3 init_R_;
  Vec3 init_t_;
};
}  // namespace lixel
