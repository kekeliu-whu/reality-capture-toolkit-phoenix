//
// Created by youyuan on 24-1-17.
//
#include <algorithm>
#include <unordered_set>
#include <utility>
#include <glog/logging.h>
#include "xmap.h"
#include "xmap_util.h"

namespace xmap {

void Xmap::mapIncremental(const PointCloudPtr& points_to_add, const V3F& view_point) {
  if (!points_to_add || points_to_add->empty()) return;

  double ts_global = static_cast<double>(points_to_add->header.stamp) * 1e-6;
  if (start_mapping_ts_ == INIT_STARTING_MAPPING_TS) start_mapping_ts_ = ts_global;

  // 维护与开机的相对时间
  auto ts_relative = static_cast<float>(ts_global - start_mapping_ts_);
  std::unordered_set<VoxelLoc> need_reconstruct_set;

  /*** 1. 迭代所有待添加点，执行地图增广 ***/
  /** 注意，这里必须触发拷贝构造，不然可能修改外部的入参导致错误 **/
  for (auto point : *points_to_add) {
    V3F point_v3f = pointType2V3F(point);
    // 1.0 若开启法向过滤，则在点无法向时：赋予视椎方向
    if (configs_.enable_normal_filter) {
      if (point.normal_x == 0 && point.normal_y == 0 && point.normal_z == 0) {
        V3F view_vec = view_point - point_v3f;
        view_vec.normalize();
        point.normal_x = view_vec.x();
        point.normal_y = view_vec.y();
        point.normal_z = view_vec.z();
      }
    }
    point.curvature = ts_relative;
    VoxelLoc small_voxelLoc = pos2VoxelLoc(point, configs_.small_voxel_size);
    // 1.1 进行small_voxel_map的增广
    auto it_small_map = small_voxel_map_.find(small_voxelLoc);

    // 1.2 生产版本使用 small_scale^3 的稠密格索引，不使用哈希。
    const IntDataType dense_index = denseCellIndex(point_v3f, small_voxelLoc, configs_);
    if (dense_index < 0) {
      LOG(WARNING) << "point fell outside its XMap small voxel";
      continue;
    }

    // 1.3 将本点添加进容器
    if (it_small_map != small_voxel_map_.end()) {
      // 如果体素已存在，则更新或添加点
      auto& value = it_small_map->second;
      const IntDataType point_index = value.index_map_[dense_index];
      const bool in_resolution = point_index >= 0;

      if (in_resolution) {
        // 在V2.0中使用策略更新点，法向由于只是过滤点，因此不更新
        const int index = point_index;
        switch (configs_.replace_points_flag) {
          case OLD_REMAIN:
            if (ts_relative < value.cloud_->points.at(index).curvature) {
              value.cloud_->points.at(index) = point;
              value.counter_ += 1;
            }
            break;
          case NEW_REPLACE:
            if (ts_relative > value.cloud_->points.at(index).curvature) {
              value.cloud_->points.at(index) = point;
              value.counter_ += 1;
            }
            break;
          case FUSION:
            // 新旧融合时，选择靠近中心的点进行替换
            const V3F grid_center = denseCellCenter(small_voxelLoc, dense_index, configs_);
            const PointType& old_point = value.cloud_->points.at(index);
            double dist_to_center_old = (grid_center - pointType2V3F(old_point)).norm();
            double dist_to_center_cur = (grid_center - point_v3f).norm();
            if (dist_to_center_cur < dist_to_center_old) {
              value.cloud_->points.at(index) = point;
              value.counter_ += 1;
            }
            break;
        }
        value.time_mark_.end_ts = ts_relative;
        if (value.counter_ > 0) need_reconstruct_set.insert(small_voxelLoc);
      } else {
        // 将点添加到体素的点云中
        value.cloud_->push_back(point);
        value.index_map_[dense_index] = static_cast<IntDataType>(value.cloud_->size() - 1);
        value.counter_ += 1;
        value.time_mark_.end_ts = ts_relative;
        need_reconstruct_set.insert(small_voxelLoc);
      }
    } else {
      // 如果体素不存在，则创建新的体素并添加到小网格地图中，并确定初次启用时间
      SmallVoxelValue value(configs_);
      TimeMark time_mark{ts_relative, ts_relative};

      value.counter_ += 1;
      value.cloud_->push_back(point);
      value.cloud_->header.stamp = points_to_add->header.stamp;
      value.index_map_[dense_index] = 0;
      value.time_mark_ = time_mark;
      need_reconstruct_set.insert(small_voxelLoc);
      small_voxel_map_.emplace(small_voxelLoc, std::move(value));
    }
  }

  /*** 2. 按需重构KD-Tree ***/
  std::vector<SmallVoxelValue*> need_reconstruct;
  need_reconstruct.reserve(need_reconstruct_set.size());
  for (const auto& voxelLoc : need_reconstruct_set) {
    auto it = small_voxel_map_.find(voxelLoc);
    if (it == small_voxel_map_.end()) {
      LOG(ERROR) << "error occured in voxel_map sync";
      // 二级告警：lru与voxel_map不同步，但可以继续跑，算法某处出错了
      continue;
    }
    need_reconstruct.push_back(&it->second);
  }

#ifdef MP_EN
#pragma omp parallel for schedule(dynamic)
#endif
  for (int i = 0; i < static_cast<int>(need_reconstruct.size()); ++i) {
    SmallVoxelValue& value = *need_reconstruct[i];
    if (value.cloud_->empty()) continue;

    const float new_rate =
        static_cast<float>(value.counter_) / static_cast<float>(value.cloud_->size());
    if (new_rate > KDTREE_UPDATE_RATIO || value.cloud_->size() < configs_.convergence_num) {
      value.kd_tree_->setInputCloud(value.cloud_);
      value.counter_ = 0;
    }
  }

  /*** 3*. 重新计算法向 ***/
  // if (configs_.enable_normal_filter) recomputeNormal(view_point);
}

void Xmap::recomputeNormal(const V3F& view_point) {
  //  for (auto& entry : small_voxel_map_) {
  //    auto& value = entry.second;
  //    auto& no_normal_vec = value.no_normal_points_;
  //    std::vector<std::pair<int, int>> replace_no_normal_vec;
  //    for (auto& it : no_normal_vec) {
  //      it.second = it.second + 1;  // 拟合次数 + 1
  //      bool re_attemp = it.second < MAX_ATTEMP;
  //      PointType& no_normal_point = value.cloud_->at(it.first);
  //      /** 1. knn搜索 **/
  //      std::vector<V3F> nearest_points;
  //      V3F search_point = pointType2V3F(no_normal_point);
  //      if (!knnSearch(search_point, view_point, nearest_points, 0) && re_attemp) {
  //        replace_no_normal_vec.push_back(it);
  //        continue;
  //      }
  //
  //      /** 2. 特征值分解 **/
  //      PlaneConstPtr plane_ptr = planeFitting(nearest_points, view_point);
  //      if (!plane_ptr->is_plane && re_attemp) {
  //        replace_no_normal_vec.push_back(it);
  //        continue;
  //      }
  //      no_normal_point.normal_x = plane_ptr->normal.x();
  //      no_normal_point.normal_y = plane_ptr->normal.y();
  //      no_normal_point.normal_z = plane_ptr->normal.z();
  //    }
  //    /** 3. 确定还有哪些点没有法向 **/
  //    replace_no_normal_vec.swap(value.no_normal_points_);
  //  }
}

void Xmap::forget(const double now_ts, const V3F& view_point) {
  if (!configs_.enable_forget) return;

  /** 1. 创建辅助向量，避免一边遍历一边删除 **/
  std::vector<std::pair<VoxelLoc, float>> voxelLocs;
  // 遍历 small_voxel_map_
  for (const auto& pair : small_voxel_map_) {
    const VoxelLoc& voxelLoc = pair.first;
    const SmallVoxelValue& voxelValue = pair.second;
    float endTs = voxelValue.time_mark_.end_ts;
    // 将 VoxelLoc 和 end_ts 插入辅助向量中
    voxelLocs.emplace_back(voxelLoc, endTs);
  }
  /** 2. 根据end_ts与格子与当前位置的相对距离，对整个格子进行遗忘 **/
  double forget_end_ts = now_ts - configs_.forget_ts;
  for (const auto& pair : voxelLocs) {
    float dist = (voxelLoc2V3F(pair.first, configs_.small_voxel_size) - view_point).norm();
    bool forget_flag = false;

    if (pair.second + start_mapping_ts_ < forget_end_ts) forget_flag = true;
    if (dist > configs_.forget_range) forget_flag = true;
    if (!forget_flag) continue;
    // TODO: solid之后可以直接delete取消保护操作，简化代码
    VoxelLoc delete_voxelLoc = pair.first;
    auto it = small_voxel_map_.find(delete_voxelLoc);

    if (it != small_voxel_map_.end()) {
      // 如果体素已存在，则直接删除本体素
      small_voxel_map_.erase(it);
    } else {
      // check: 如果体素不存在，则系统发生了错误
      LOG(ERROR) << "Voxel of cache points not exist!";
    }
  }
}

}  // namespace xmap
