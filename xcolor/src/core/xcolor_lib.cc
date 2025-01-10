
#include <CGAL/AABB_traits_3.h>
#include <CGAL/AABB_tree.h>
#include <CGAL/AABB_triangle_primitive_3.h>
#include <CGAL/Simple_cartesian.h>
#include <absl/container/flat_hash_map.h>
#include <absl/container/flat_hash_set.h>
#include <pcl/io/pcd_io.h>
#include <pcl/range_image/range_image_planar.h>
#include <opencv2/opencv.hpp>

#include "common/types.h"
#include "core/xcolor_lib.h"
#include "migration/utils.h"

typedef CGAL::Simple_cartesian<double> K;
typedef K::Point_3 Point;
typedef K::Segment_3 Segment;
typedef K::Iso_cuboid_3 Cubic;
typedef std::list<Cubic>::iterator Iterator;
typedef CGAL::AABB_triangle_primitive_3<K, Iterator> Primitive;
typedef CGAL::AABB_traits_3<K, Primitive> AABB_triangle_traits;
typedef CGAL::AABB_tree<AABB_triangle_traits> Tree;
typedef Tree::Primitive_id Primitive_id;

// Hash value
namespace std {
template <>
struct hash<Eigen::Vector3i> {
  int64_t operator()(const Eigen::Vector3i &s) const {
    using std::hash;
    using std::size_t;
#define HASH_P 116101
#define MAX_N 10000000000
    return (((s.z() * HASH_P) % MAX_N + s.y()) * HASH_P) % MAX_N + s.x();
  }
};

}  // namespace std

struct VoxelVisibleInfo {
  std::shared_ptr<std::vector<int>> point_indices;
  std::shared_ptr<std::vector<int>> visible_image_indices;
  Eigen::Vector3d center;

  VoxelVisibleInfo() {
    point_indices         = std::make_shared<std::vector<int>>();
    visible_image_indices = std::make_shared<std::vector<int>>();  // todo rename visualb_
  }
};

class PointCloudRayCaster {
 public:
  PointCloudRayCaster(double resolution) { resolution_ = resolution; }

  void AddPointCloud(const pcl::PointCloud<pcl::PointXYZRGB> &cloud) {
    // collect voxel_boxes_cache_ and voxel_map_'s point_indices
    voxel_boxes_cache_.clear();
    for (int i = 0; i < cloud.size(); ++i) {
      Eigen::Vector3i key = PointToVoxel(cloud[i].getVector3fMap().cast<double>());
      if (voxel_map_.find(key) == voxel_map_.end()) {
        auto box = VoxelToBox(key);
        voxel_boxes_cache_.push_back(Cubic(box.min().x(), box.min().y(), box.min().z(), box.max().x(), box.max().y(), box.max().z()));
      }
      voxel_map_[key].point_indices->push_back(i);
    }
    tree_.reset(new Tree(voxel_boxes_cache_.begin(), voxel_boxes_cache_.end()));
    LOG(INFO) << "Build Voxelmap and AABBTree done: voxel count: " << voxel_map_.size() << " point count: " << cloud.size();

    // compute centroid as the voxel center
    for (auto &[_, voxel] : voxel_map_) {
      Eigen::Vector3d coord_sum = Eigen::Vector3d::Zero();
      for (auto &idx : *voxel.point_indices) {
        coord_sum += cloud[idx].getVector3fMap().cast<double>();
      }
      voxel.center = coord_sum / voxel.point_indices->size();
    }
    LOG(INFO) << "Compute voxel center done.";
  }

  void PerformRayCasting(const std::vector<Image> &images) {
    LOG(INFO) << "Perform Ray Casting...";
    std::atomic<int> process_num(0);
#pragma omp parallel for
    for (int image_idx = 0; image_idx < images.size(); ++image_idx) {
      Image image             = images[image_idx];
      Eigen::Vector3d cam_pos = colmap::Inverse(image.pose).translation;

      std::map<double, Eigen::Vector3i, std::greater<double>> voxel_candidate_map;
      for (auto &[voxel_key, ignored] : voxel_map_) {
        double dist = (voxel_map_[voxel_key].center - cam_pos).norm();
        if (dist < 60) {  // todo kk magic number
          voxel_candidate_map[dist] = voxel_key;
        }
      }

      absl::flat_hash_set<Eigen::Vector3i> visited_voxel_set;
      absl::flat_hash_set<Eigen::Vector3i> visible_voxel_set;
      for (auto &[ignored, voxel_key] : voxel_candidate_map) {
        // use visited_voxel_set to reduce ray casting count
        if (visited_voxel_set.find(voxel_key) != visited_voxel_set.end()) {
          continue;
        }
        std::vector<Eigen::Vector3i> visited_voxels;
        RayCastingCamCenterToTargetPoint(image, voxel_map_[voxel_key].center, visited_voxels, visible_voxel_set);
        for (auto &voxel : visited_voxels) {
          visited_voxel_set.insert(voxel);
        }
      }

#pragma omp critical
      {
        LOG(INFO) << "Progress: " << ++process_num << " / " << images.size();
        for (auto &voxel : visible_voxel_set) {
          voxel_map_[voxel].visible_image_indices->push_back(image_idx);
        }
      }
    }
  }

  const auto &GetRayCastingResult() const { return voxel_map_; }

 private:
  Eigen::Vector3i PointToVoxel(const Eigen::Vector3d &point) const { return (point / resolution_).array().floor().cast<int>(); }

  Eigen::AlignedBox3d VoxelToBox(const Eigen::Vector3i &voxel) const {
    return {voxel.cast<double>() * resolution_, (voxel.cast<double>() + Eigen::Vector3d::Ones()) * resolution_};
  }

  void RayCastingCamCenterToTargetPoint(const Image &image, const Eigen::Vector3d &target_point, std::vector<Eigen::Vector3i> &visited_voxel_indices,
                                        absl::flat_hash_set<Eigen::Vector3i> &visible_voxel_indices) const {
    Eigen::Vector3d pt_in_cam = image.pose * target_point;
    if (pt_in_cam.z() < 0) {
      return;
    }

    Eigen::Vector3d from = colmap::Inverse(image.pose).translation;
    Eigen::Vector3d to   = target_point;

    Segment segment_query(Point(from.x(), from.y(), from.z()), Point(to.x(), to.y(), to.z()));
    std::list<Primitive_id> primitives;
    tree_->all_intersected_primitives(segment_query, std::back_inserter(primitives));

    if (primitives.empty()) {
      return;
    }

    std::map<double, Primitive_id> primitives_ordered;
    for (auto &prim : primitives) {
      auto box = prim->bbox();
      Eigen::Vector3d center{(box.xmin() + box.xmax()) / 2, (box.ymin() + box.ymax()) / 2, (box.zmin() + box.zmax()) / 2};
      Eigen::Vector3i voxel_id = PointToVoxel(center);

      double dist              = (voxel_map_.at(voxel_id).center - from).norm();
      primitives_ordered[dist] = prim;

      visited_voxel_indices.push_back(voxel_id);
    }

    double nearest_dist = primitives_ordered.begin()->first;
    for (auto &[dist, prim] : primitives_ordered) {
      if (dist - nearest_dist > 0.1) {  // todo kk
        break;
      }
      auto box = prim->bbox();
      Eigen::Vector3d center{(box.xmin() + box.xmax()) / 2, (box.ymin() + box.ymax()) / 2, (box.zmin() + box.zmax()) / 2};
      Eigen::Vector3i voxel_id = PointToVoxel(center);
      visible_voxel_indices.insert(voxel_id);
    }
  }

 private:
  absl::flat_hash_map<Eigen::Vector3i, VoxelVisibleInfo> voxel_map_;
  std::shared_ptr<Tree> tree_;
  double resolution_;
  std::list<Cubic> voxel_boxes_cache_;  // cache voxel bounding boxes for AABB tree
};

void PerformXColor(const pcl::PointCloud<pcl::PointXYZRGB> &cloud, const std::vector<Image> &images, std::string output_path) {
  int min_cand_num                   = 3;
  int color_inlier_max_num           = 30;
  int color_inlier_threshold         = 60;
  int range_inlier                   = 60;
  double ray_caster_voxel_resolution = 0.08;

  PrintMemoryUsage();
  pcl::PointCloud<pcl::PointXYZRGBL> cloud_rgb;
  pcl::copyPointCloud(cloud, cloud_rgb);

  PrintMemoryUsage();
  PointCloudRayCaster ray_caster(ray_caster_voxel_resolution);
  ray_caster.AddPointCloud(cloud);
  PrintMemoryUsage();
  ray_caster.PerformRayCasting(images);

  LOG(INFO) << "Start xcolor...";
  auto voxel_map = ray_caster.GetRayCastingResult();
  std::vector<Eigen::Vector3i> keys;
  for (const auto &pair : voxel_map) {
    keys.push_back(pair.first);
  }

#pragma omp parallel for
  for (int k = 0; k < keys.size(); ++k) {
    const auto &voxel_ray_casting = voxel_map.at(keys[k]);
    for (int point_idx : *voxel_ray_casting.point_indices) {
      // compute color candidates by in FOV
      std::vector<std::pair<double, cv::Vec3b>> color_candidates;
      for (int image_idx : *voxel_ray_casting.visible_image_indices) {
        Image image       = images[image_idx];
        Eigen::Matrix3d K = image.camera.CalibrationMatrix();

        Eigen::Vector3d pt_in_cam = image.pose * cloud_rgb[point_idx].getVector3fMap().cast<double>();
        if (pt_in_cam.z() < 0) {
          continue;  // todo kk handle fish eye FOV over 180°
        }
        Eigen::Vector2d pixel = image.camera.ImgFromCam((pt_in_cam / pt_in_cam.z()).head<2>());
        if (pixel.x() < 0 || pixel.x() >= image.image.cols || pixel.y() < 0 || pixel.y() >= image.image.rows) {
          continue;
        }
        cv::Vec3b color = image.image.at<cv::Vec3b>(pixel.y(), pixel.x());

        double distance = pt_in_cam.norm();
        color_candidates.push_back({distance, color});
      }

      // sort by distance
      if (color_candidates.size() > color_inlier_max_num) {
        std::sort(color_candidates.begin(), color_candidates.end(), [](auto &a, auto &b) { return a.first < b.first; });
        color_candidates.erase(color_candidates.begin() + color_inlier_max_num, color_candidates.end());
      }

      if (color_candidates.empty()) {
        continue;
      } else if (color_candidates.size() > min_cand_num) {
        for (int k = 0; k < 3; ++k) {
          std::vector<uint8_t> color_channel_candidates;
          color_channel_candidates.reserve(color_candidates.size());
          for (int j = 0; j < color_candidates.size(); ++j) {
            color_channel_candidates.push_back(color_candidates[j].second[k]);
          }

          int mid_idx = color_channel_candidates.size() / 2;
          std::nth_element(color_channel_candidates.begin(), color_channel_candidates.begin() + mid_idx, color_channel_candidates.end());
          uint8_t mid_color = color_channel_candidates[mid_idx];

          int color_sum = 0;
          int color_num = 0;
          for (int j = 0; j < color_channel_candidates.size(); ++j) {
            if (std::abs((int)color_channel_candidates[j] - (int)mid_color) < color_inlier_threshold) {
              // todo kk by distance
              color_sum += color_channel_candidates[j];
              color_num += 1;
            }
          }

          cloud_rgb[point_idx].getBGRVector3cMap()[k] = color_sum / color_num;
          cloud_rgb[point_idx].label                  = voxel_ray_casting.visible_image_indices->size();
        }
      } else {
        CHECK_GT(color_candidates.size(), 0) << point_idx;
        // evaluate by mean value
        cv::Vec3i color = {0, 0, 0};
        for (int j = 0; j < color_candidates.size(); ++j) {
          // todo kk by distance
          color += color_candidates[j].second;
        }
        cloud_rgb[point_idx].b     = color[0] / (int)color_candidates.size();
        cloud_rgb[point_idx].g     = color[1] / (int)color_candidates.size();
        cloud_rgb[point_idx].r     = color[2] / (int)color_candidates.size();
        cloud_rgb[point_idx].label = voxel_ray_casting.visible_image_indices->size();
      }
    }
  }

  LOG(INFO) << "Saving point cloud to " << output_path + "/xcolor.pcd";
  pcl::io::savePCDFileBinary(output_path + "/xcolor.pcd", cloud_rgb);

  LOG(INFO) << "Finish xcolor.";
}
