
#include <glog/logging.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <thread>
#include <vector>

#include "io/read_write.h"
#include "utils.h"

void PcaEstimateNormal(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                       const std::vector<Eigen::Vector3f> &centers,
                       int k,
                       double downsample_voxel_size,
                       std::vector<Eigen::Vector3f> &normals) {
  normals.resize(cloud->size());

  DLOG(INFO) << "Building kdtree for normal estimation...";
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  DownsamplePointCloud(cloud, cloud_downsampled, downsample_voxel_size);
  tree->setInputCloud(cloud_downsampled);

  DLOG(INFO) << "Estimating normals...";
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
  pcl::VoxelGrid<pcl::PointXYZI> filter;
  filter.setInputCloud(cloud_in);
  filter.setLeafSize(voxel_size, voxel_size, voxel_size);
  filter.filter(*cloud_out);
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
  for (auto &submap : submaps) {
    pcl::VoxelGrid<pcl::PointXYZI> voxel_grid;
    voxel_grid.setInputCloud(submap.cloud);
    voxel_grid.setLeafSize(voxel_size, voxel_size, voxel_size);
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered(
        new pcl::PointCloud<pcl::PointXYZI>);
    voxel_grid.filter(*cloud_filtered);
    // DLOG(INFO) << "Downsample submap from " << submap.cloud->size() << " to " << cloud_filtered->size() << " points.";
    submap.cloud = cloud_filtered;
  }

  DLOG(INFO) << "Downsample " << submaps.size() << " submaps done.";
}
