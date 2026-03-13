
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <spdlog/spdlog.h>
#include <thread>
#include <vector>

#include "io/read_write.h"
#include "utils.h"

#define HASH_P 116101
#define MAX_N 10000000000

class VoxelLoc {
 public:
  int64_t x, y, z;

  VoxelLoc(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0)
      : x(vx), y(vy), z(vz) {}

  bool operator==(const VoxelLoc &other) const {
    return (x == other.x && y == other.y && z == other.z);
  }
};

// Hash value
namespace std {
template <>
struct hash<VoxelLoc> {
  int64_t operator()(const VoxelLoc &s) const {
    using std::hash;
    using std::size_t;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

struct M_POINT {
  Eigen::Vector3d center;
  int count = 0;
};

template <typename PointType>
void DownsamplePointCloudInternal(const pcl::PointCloud<PointType> &cloud_in,
                                  pcl::PointCloud<PointType> &cloud_out,
                                  double voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  std::unordered_map<VoxelLoc, M_POINT> feat_map;

  for (int i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().template cast<double>();
    int loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.center += p_c;
      iter->second.count++;
    } else {
      M_POINT p;
      p.center           = p_c;
      p.count            = 1;
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  int i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.center.cast<float>() / iter->second.count;
    i++;
  }
}

void PcaEstimateNormal(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                       const std::vector<Eigen::Vector3f> &centers,
                       int k,
                       double downsample_voxel_size,
                       std::vector<Eigen::Vector3f> &normals) {
  normals.resize(cloud->size());

  spdlog::info("Building kdtree for normal estimation...");
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  DownsamplePointCloud(cloud, cloud_downsampled, downsample_voxel_size);
  tree->setInputCloud(cloud_downsampled);

  spdlog::info("Estimating normals...");
#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(k);
    std::vector<float> k_sqr_distances(k);

    if (tree->nearestKSearch(cloud->at(i), k, k_indices, k_sqr_distances) <= 0) {
      continue;
    }

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int j : k_indices) {
      centroid += cloud_downsampled->points[j].getVector3fMap().cast<double>();
    }
    centroid /= k_indices.size();

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int j : k_indices) {
      Eigen::Vector3d neighbor = cloud_downsampled->points[j].getVector3fMap().cast<double>();
      Eigen::Vector3d cp       = neighbor - centroid;
      covariance += cp * cp.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
    Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();

    normals[i] = eigenvectors.col(0).cast<float>();
    if ((cloud->at(i).getVector3fMap() - centers[i]).dot(normals[i]) < 0) {
      normals[i] = -normals[i];
    }
  }
}

void DownsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud_in,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_out,
                          double voxel_size) {
  DownsamplePointCloudInternal<pcl::PointXYZI>(*cloud_in, *cloud_out, voxel_size);
}

void LoadFullPointCloud(const std::string &project_input_path,
                        pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                        std::vector<Eigen::Vector3f> &centers) {
  std::vector<TimestampedPointCloud> submap_list;
  LoadSubmapList(project_input_path, submap_list, 0);

  for (auto &submap : submap_list) {
    for (auto &p : *submap.cloud) {
      pcl::PointXYZI np;
      np.getVector3fMap() = (submap.pose * p.getVector3fMap().cast<double>()).cast<float>();
      np.intensity        = p.intensity;

      cloud->points.push_back(np);
      centers.push_back(submap.pose.translation().cast<float>());
    }
  }
}

void DownsampleSubmaps(std::vector<TimestampedPointCloud> &submaps,
                       double voxel_size) {
#pragma omp parallel for
  for (int i = 0; i < submaps.size(); ++i) {
    auto &submap = submaps[i];
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
    voxel_grid.setInputCloud(submap.cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZI>);
    voxel_grid.filter(*cloud_filtered);
    // DLOG(INFO) << "Downsample submap from " << submap.cloud->size() << " to " << cloud_filtered->size() << " points.";
    submap.cloud = cloud_filtered;
  }

  spdlog::info("Downsample {} submaps done.", submaps.size());
}
