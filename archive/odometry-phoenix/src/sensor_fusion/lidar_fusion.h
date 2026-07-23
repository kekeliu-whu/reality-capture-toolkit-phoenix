//
// Created by youyuan on 24-2-18.
//

#pragma once

#include "base_fusion.h"
#include "common/math_utils.h"
#include "xmap.h"
namespace lixel
{

constexpr double S_MIN = 0.1;

constexpr int WARNING_POINTS_NUM = 30;
constexpr int ERROR_POINTS_NUM = 5;

struct LiDARMeasTemp
{
  int index;
  Vec3 p_I;
  Vec3 normal;
  double residual;
  double var;
  double weight;
  bool useful = false;
};

class LiDARFusion : public BaseFusion
{
 public:
  LiDARFusion(double k_for_adaptive_search);
  void setXmap(std::shared_ptr<xmap::Xmap>& xmap_ptr);
  void setLidarMeas(
      const PointCloudXYZINormal::Ptr& surf_pcl,
      const PointCloudXYZINormal::Ptr& map_pcl,
      const PointCloud::Ptr& undistort_pcl);
  bool getUpdateFrame(PointCloudXYZINormal::Ptr& update_frame);
  bool getPublishFrame(PointCloud::Ptr& publish_frame);
  void calculateMeas(VecX& residual, SparseMat& H, double& R) override;
  const AttributeJacobi& getAttributeJacobi() const;

  KFState::ConstPtr temp_state_;

 private:
  AttributeJacobi attributeJacobi_;
  std::shared_ptr<xmap::Xmap> xmap_ptr_;
  PointCloudXYZINormal::Ptr sw_lidar_surf_[WINDOW_SIZE];
  PointCloudXYZINormal::Ptr sw_lidar_map_[WINDOW_SIZE];
  PointCloud::Ptr sw_lidar_undistort_[WINDOW_SIZE];
  double k_for_adaptive_search_;  // outlier judgement: residual > k_for_adaptive_search * dist + S_MIN
  void calculateAttrJacobi(const std::vector<LiDARMeasTemp>& meas_vec);
};
}  // namespace lixel
