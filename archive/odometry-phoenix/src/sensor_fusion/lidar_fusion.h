//
// Created by youyuan on 24-2-18.
//

#pragma once

#include "base_fusion.h"
#include "common/math_utils.h"
#include "parameters.h"
#include "xmap.h"
namespace lixel
{

constexpr int WARNING_POINTS_NUM = 100;
constexpr int ERROR_POINTS_NUM = 10;

enum class LiDARMatchStatus : int32_t
{
  KNN_FAIL = 0,
  OUTLIER = 1,
  NO_PLANE = 2,
  ACTIVE = 3
};

struct LiDARMatchCache
{
  LiDARMatchStatus status = LiDARMatchStatus::ACTIVE;
  xmap::PlaneConstPtr plane;
};

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
  explicit LiDARFusion(const IESKFParam& param);
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
  AttributeJacobi attributeJacobi_{};
  std::shared_ptr<xmap::Xmap> xmap_ptr_;
  std::vector<PointCloudXYZINormal::Ptr> sw_lidar_surf_;
  std::vector<PointCloudXYZINormal::Ptr> sw_lidar_map_;
  std::vector<PointCloud::Ptr> sw_lidar_undistort_;
  std::vector<std::vector<LiDARMatchCache>> sw_match_cache_;
  int window_size_;
  int reset_window_size_;
  bool faster_model_;
  bool refresh_matches_ = true;
  double knn_search_slope_;
  double knn_search_min_dist_;
  void calculateAttrJacobi(const std::vector<LiDARMeasTemp>& meas_vec);
};
}  // namespace lixel
