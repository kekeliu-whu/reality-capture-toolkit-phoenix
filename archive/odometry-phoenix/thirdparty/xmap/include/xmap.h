// 涉及Xmap的功能定义与实现
// 地图定义：大小格GridlMap，大格GridMap用于地图动态加载，小格GridMap用于实际的地图增广与搜索
// 接口：支持KNN搜索，半径搜索，地图增广，地图动态加载与删除，地图遗忘，法向过滤作为配置无需外部调用
// 设计文档：https://pecivkvtit.feishu.cn/docx/Vq1OdiUmjoq3D5xa9IbczkMhnsb

#pragma once

#include <list>
#include <unordered_map>
#include <unordered_set>

#include "small_voxel_value.h"
#include "voxel_loc.h"
#include "xmap_types.hpp"

namespace xmap {

class Xmap {
 public:
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
   * 动态地图加载功能接口
   * 依次调用地图落盘与加载，并记录动态加载的位置与体素ID
   * @param view_point 当前位置
   * @return
   */
  void dynamicSaveAndLoad(const V3F& view_point);

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
  // clang-format off
  std::unordered_map<VoxelLoc, SmallVoxelValue> small_voxel_map_;               // Xmap的实际地图 - 搜索载体
  std::unordered_map<VoxelLoc, std::unordered_set<VoxelLoc>> large_voxel_map_;  // Xmap的分块地图 - 加载载体 - 开启动态加载才会有值

  Configs configs_;                                   // 地图专属配置，与外部无耦合
  double start_mapping_ts_ = -1;                      // 开始建图的时间
  V3F last_dynamic_view_point_ = V3F::Zero();         // 上次动态加载的位置
  VoxelLoc last_dyanamic_large_voxel_{0, 0, 0};       // 上次动态加载的体素 -- 动态规划思想 -- 防止多次计算
  // clang-format on

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

  /**
   * 重新计算每个点的法向
   * @param view_point 当前位置
   */
  void recomputeNormal(const V3F& view_point);

  /**
   * 根据view_point, 把磁盘中即将进入视野的大格点云加载到内存之中
   * @param view_point 当前位置
   * @return 是否成功
   */
  void dynamicLoad(const V3F& view_point);

  /**
   * 根据view_point, 把在视野外的大格点云保存到磁盘之中
   * @param view_point 当前位置
   * @return 是否成功
   */
  void dynamicSave(const V3F& view_point);

  /**
   * 检验大小格子中的元素是否同步
   * @return 同步返回true，不同步返回false
   */
  bool checkGridDataSync();

  /**
   * 返回本VoxelLoc是否在MapSize内
   * @param ref_voxel_loc 待确定的大格子VoxelLoc
   * @param center_voxel_loc 当前Map中心的大格子VoxelLoc
   * @return
   */
  inline bool inMap(const VoxelLoc& ref_voxel_loc, const VoxelLoc& center_voxel_loc) {
    VoxelLoc minus = center_voxel_loc - ref_voxel_loc;
    return abs(minus.x_) <= configs_.voxel_diff && abs(minus.y_) <= configs_.voxel_diff &&
           abs(minus.z_) <= configs_.voxel_diff;
  }

  friend class XmapTest;
};
}  // namespace xmap
