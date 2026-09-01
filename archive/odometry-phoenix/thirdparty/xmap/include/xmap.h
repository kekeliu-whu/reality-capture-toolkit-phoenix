// 涉及Xmap的功能定义与实现
// 地图定义：小格GridMap用于地图增广与搜索
// 接口：支持KNN搜索、半径搜索、地图增广、地图遗忘，法向过滤作为配置无需外部调用
// 设计文档：https://pecivkvtit.feishu.cn/docx/Vq1OdiUmjoq3D5xa9IbczkMhnsb

#pragma once

#include <unordered_map>

#include "small_voxel_value.h"
#include "voxel_loc.h"
#include "xmap_types.hpp"

namespace xmap {

class Xmap {
 public:
  explicit Xmap(const Configs& configs);
  explicit Xmap(const std::string& path);
  ~Xmap();

  /**
   * 地图增广功能接口
   * @param points_to_add 待加到地图中的点
   * @param view_point 当前位置，会用于法向的重新计算
   */
  void mapIncremental(const PointCloudPtr& points_to_add, const V3F& view_point);

  /**
   * knn搜索功能接口
   * @param point_search 待搜索点
   * @param view_point 当前位置
   * @param nearest_points point_search最近的knn_min_points个点
   * @param ts_absolute 滤波器当前时间戳，更新active time
   * @return 是否搜索成功
   */
  bool knnSearch(
      const V3F& point_search,
      const V3F& view_point,
      std::vector<V3F>& nearest_points,
      double ts_absolute);

  /** KNN搜索并拟合查询点附近的平面。 */
  bool knnSearch(
      const V3F& point_search,
      const V3F& view_point,
      double ts_absolute,
      PlaneConstPtr& plane);

  /**
   * 半径搜索功能接口
   * @param point_search 待搜索点
   * @param view_point 当前位置
   * @param nearest_points 搜索半径knn_distance_limit内的点
   * @param ts_absolute 滤波器当前时间戳，更新active time
   * @return 是否搜索成功
   */
  bool radiusSearch(
      const V3F& point_search,
      const V3F& view_point,
      std::vector<V3F>& nearest_points,
      double ts_absolute);

  /**
   * surfel搜索功能接口，暂时无用
   * 未来可能每个小格子对应一个surfel
   * 类似方法可参考：ESM, VoxelMap && VoxelMap++, LiTAMIN2
   * @param point_search 待搜索点
   * @param plane 搜索的平面
   * @return 搜索是否成功
   */
  bool planeSearch(const V3F& point_search, Plane& plane);

  /**
   * 地图遗忘功能接口
   * @param now_ts 当前时间
   * @param view_point 当前位置
   */
  void forget(const double now_ts, const V3F& view_point);

  /**
   * 获取xmap的配置
   * @return 配置对象
   */
  const Configs& getConfigs() const;

  /**
   * 迭代统计xmap中地图点的数目
   * @return 总点数
   */
  int pointSize() const;

  /**
   * 迭代统计xmap中预分配的点云容量
   * @return 总容量
   */
  int capacity() const;

  /**
   * 重构所有的KDTree，特别耗时，自测使用
   */
  void reBuildKDTree();

  /**
   * 返回Xmap中的点云
   */
  PointCloudConstPtr getMapPointCloud() const;

 private:
  std::unordered_map<VoxelLoc, SmallVoxelValue> small_voxel_map_;
  Configs configs_;
  double start_mapping_ts_ = -1;

  /**
   * 以point为中心，判断哪些体素需要执行knn搜索 || radius搜索
   * @param point 待搜索点
   * @return 待搜索体素的空间索引
   */
  std::vector<VoxelLoc> generateSearchVoxelKey(const V3F& point);

  /**
   * 加载配置到configs_
   * @param path 配置文件路径
   */
  void loadConfigsFromYaml(const std::string& path);

  /** Calculate derived fields and validate configuration. */
  void finalizeConfigs();

  /** Shared KNN implementation retained for tests and the compatibility API. */
  bool collectNearestPoints(
      const V3F& point_search,
      const V3F& view_point,
      std::vector<V3F>& nearest_points,
      double ts_absolute);

  /**
   * 重新计算每个点的法向
   * @param view_point 当前位置
   */
  void recomputeNormal(const V3F& view_point);

  friend class XmapTest;
};
}  // namespace xmap
