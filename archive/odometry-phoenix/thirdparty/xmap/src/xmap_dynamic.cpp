//
// Created by youyuan on 24-1-17.
//

#include <glog/logging.h>
#ifdef __linux__
#include <malloc.h>
#endif
#include "xmap.h"
#include "xmap_util.h"

namespace xmap {

void Xmap::dynamicSaveAndLoad(const V3F& view_point) {
  if (!configs_.enable_dynamic) return;

  VoxelLoc large_voxel_center_voxelLoc = pos2VoxelLoc(view_point, configs_.large_voxel_size);
  double move_dist = (last_dynamic_view_point_ - view_point).norm();
  if (large_voxel_center_voxelLoc == last_dyanamic_large_voxel_ ||
      move_dist < configs_.large_voxel_size / 3) {
    return;
  }
  last_dyanamic_large_voxel_ = large_voxel_center_voxelLoc;
  last_dynamic_view_point_ = view_point;

  dynamicSave(view_point);
  dynamicLoad(view_point);
}

void Xmap::dynamicLoad(const V3F& view_point) {
  TicToc load_total;
  VoxelLoc large_voxel_center_voxelLoc = pos2VoxelLoc(view_point, configs_.large_voxel_size);
  /*** 1. 遍历获得path下的所有pcd文件名，并获得VoxelLoc ***/
  std::vector<std::pair<VoxelLoc, std::string>> all_pcd_vec;
  TicToc ts1;
  for (const auto& entry : std::filesystem::directory_iterator(configs_.pcd_path)) {
    if (entry.path().extension() == ".data") {
      std::string file_name = entry.path().filename().string();

      std::vector<IntDataType> numbers;
      std::stringstream ss(file_name);
      std::string token;
      while (std::getline(ss, token, '_')) {
        try {
          int number = std::stoi(token);
          numbers.push_back(number);
        } catch (const std::exception& e) {
          // 处理无法转换为整数的情况
          std::cout << "Error parsing integer: " << token << std::endl;
        }
      }
      VoxelLoc voxelLoc(numbers[0], numbers[1], numbers[2]);
      all_pcd_vec.emplace_back(voxelLoc, file_name);
    }
  }
  LOG(INFO) << strprintf("loadT1: %.3f", ts1.Toc());

  /*** 2. 根据VoxelLoc，判断哪些文件需要加载入voxel_map ***/
  TicToc ts2;
  std::vector<std::pair<VoxelLoc, std::string>> need_load_vec;
  for (auto& entry : all_pcd_vec) {
    if (inMap(entry.first, large_voxel_center_voxelLoc)) {
      need_load_vec.push_back(entry);

      SmallVoxelValue value(configs_);
      TimeMark time_mark{0, 0};
      value.is_dynamic_IO = true;
      value.time_mark_ = time_mark;
      small_voxel_map_.insert(std::make_pair(entry.first, value));
    }
  }
  LOG(INFO) << strprintf("loadT2: %.3f", ts2.Toc());

  // 3，4均可开启多线程
  auto func = [this](
                  const std::vector<std::pair<VoxelLoc, std::string>>& need_load_vec,
                  const V3F& view_point) {
    TicToc ts3;
    /*** 3. 根据文件名，加载pcd载入voxel_map，并构建KD-Tree ***/
    double total_load = 0.0;
    double total_incremental = 0.0;
    PointCloudPtr cloud(new PointCloud);
    for (auto& entry : need_load_vec) {
      TicToc start_load;

      std::vector<pcl::PointXYZINormal, Eigen::aligned_allocator<pcl::PointXYZINormal>> data =
          readPointCloudFromFile(configs_.pcd_path + entry.second);
      cloud->points.insert(cloud->points.end(), data.begin(), data.end());
      total_load += start_load.Toc();
    }
    TicToc map_incremental_ts;
    mapIncremental(cloud, view_point);
    total_incremental += map_incremental_ts.Toc();
    LOG(INFO) << strprintf("loadT3: %.3f", ts3.Toc());

    /*** 4. 加载完成，重置is_dynamic_IO ***/
    for (auto& entry : need_load_vec) {
      auto it = small_voxel_map_.find(entry.first);
      if (it == small_voxel_map_.end()) {
        LOG(ERROR) << "small_voxel:" << it->first;
        LOG(ERROR) << "small_voxel_map_ dosen't have loaded!";
        // 二级告警：small_voxel_map_并没有成功加载，但可以继续跑，算法某处出错了
        continue;
      } else {
        it->second.is_dynamic_IO = false;
      }
    }
    LOG(INFO) << strprintf("loadT3: %.3f", ts3.Toc());

    /*** 5. 删除点云文件 ***/
    TicToc ts4;
    for (auto& entry : need_load_vec) {
      std::string filename = configs_.pcd_path + entry.second;
      try {
        std::filesystem::remove(filename);
      } catch (const std::filesystem::filesystem_error& e) {
        LOG(WARNING) << "delete file faied:" << filename;
      }
    }
    LOG(INFO) << strprintf("loadFileT: %.3f", total_load);
    LOG(INFO) << strprintf("incrementalT: %.3f", total_incremental);
    LOG(INFO) << strprintf("loadT4: %.3f", ts4.Toc());
  };

  func(need_load_vec, view_point);

  if (configs_.enable_dynamic_backend){
    std::thread myThread(func, need_load_vec, view_point);
    myThread.detach();
  }
  else
    func(need_load_vec, view_point);
}

void Xmap::dynamicSave(const V3F& view_point) {
  // checkGridDataSync();
  TicToc total_save_t;
  VoxelLoc large_voxel_center_voxelLoc = pos2VoxelLoc(view_point, configs_.large_voxel_size);
  std::vector<std::pair<VoxelLoc, std::vector<VoxelLoc>>> erase_voxelloc;
  TicToc ts12;
  for (auto& it : large_voxel_map_) {
    /*** 1. 统计哪些格子需要落盘，并设置其is_dynamic_IO为true ***/
    if (inMap(it.first, large_voxel_center_voxelLoc)) continue;
    std::vector<VoxelLoc> erase_small_voxelloc;
    for (auto& small_voxel : it.second) {
      auto it_small = small_voxel_map_.find(small_voxel);
      if (it_small == small_voxel_map_.end()) {
        LOG(ERROR) << "small_voxel:" << small_voxel;
        LOG(ERROR) << "small_voxel_map_ not sync with large_voxel_map_!";
        // 二级告警：large_voxel_map_与small_voxel_map_不同步，但可以继续跑，算法某处出错了
        continue;
      } else {
        it_small->second.is_dynamic_IO = true;
        erase_small_voxelloc.push_back(small_voxel);
      }
    }
    erase_voxelloc.emplace_back(it.first, erase_small_voxelloc);
  }

  auto func = [this](const std::vector<std::pair<VoxelLoc, std::vector<VoxelLoc>>>& erase_voxelloc) {
    /*** 2. 汇聚需要落盘的大格子内的所有点云并落盘***/
    for (auto& voxel_pair : erase_voxelloc) {
      PointCloudPtr cloud_(new PointCloud);
      for (auto& small_voxel : voxel_pair.second) {
        auto it_small = small_voxel_map_.find(small_voxel);
        if (it_small == small_voxel_map_.end()) {
          LOG(ERROR) << "small_voxel:" << small_voxel;
          LOG(ERROR) << "small_voxel_map_ not sync with large_voxel_map_!";
          // 二级告警：large_voxel_map_与small_voxel_map_不同步，但可以继续跑，算法某处出错了
          continue;
        }
        PointCloudPtr& small_voxel_cloud_ = it_small->second.cloud_;
        cloud_->points.insert(
            cloud_->points.end(),
            small_voxel_cloud_->points.begin(),
            small_voxel_cloud_->points.end());
        cloud_->height = 1;
        cloud_->width = cloud_->size();
        std::string file_name = voxel_pair.first.toString() + ".data";
        TicToc start_save;
        if (!cloud_->empty()) writePointCloudToFile(cloud_->points, configs_.pcd_path + file_name);
      }
    }

    /*** 3. 删除容器中与这些格子关联的数据 ***/
    for (auto& voxel_pair : erase_voxelloc) {
      large_voxel_map_.erase(voxel_pair.first);
      for (auto& small_voxel : voxel_pair.second) {
        small_voxel_map_.erase(small_voxel);
      }
    }
    #ifdef __linux__
    malloc_trim(0);
    #endif
  };

  if (configs_.enable_dynamic_backend) {
    std::thread myThread(func, erase_voxelloc);
    myThread.detach();
  }
  else
    func(erase_voxelloc);
}

}  // namespace xmap