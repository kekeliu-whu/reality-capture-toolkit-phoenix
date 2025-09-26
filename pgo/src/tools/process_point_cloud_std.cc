
#include <glog/logging.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <pcl/filters/impl/fast_bilateral.hpp>
#include <pcl/filters/impl/fast_bilateral_omp.hpp>
#include <thread>
#include <vector>

#include "map/utils.h"

DEFINE_string(project_input_path, "D:/BaiduNetdiskDownload/2024-12-04-11-28-44-SHAREUAV-S20", "Input project path");
DEFINE_string(project_output_path, "D:/BaiduNetdiskDownload/2024-12-04-11-28-44-SHAREUAV-S20", "Output project path");

static constexpr double kDownsampleVoxelSize    = 0.05;
static constexpr int kNearestNeighbors          = 15;
static constexpr int kSmoothMaxNearestNeighbors = 100;
static constexpr double kSmoothMaxSearchRadius  = 0.3;
static constexpr double kSmoothSigmaD           = 0.05;
static constexpr double kSmoothSigmaN           = 0.05;

void SavePointCloud(const std::string &filename, const pcl::PointCloud<pcl::PointXYZI> &cloud, const std::vector<Eigen::Vector3f> &normals) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud.size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud.points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    cloud_out->points.push_back(np);
  }
}

void SmoothPointCloud(const std::vector<Eigen::Vector3f> &normals,
                      int kNearestNeighbors,
                      double max_search_radius,
                      double sigma_d,
                      double sigma_n,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(cloud);

#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(kNearestNeighbors);
    std::vector<float> k_sqr_distances(kNearestNeighbors);

    if (tree->radiusSearch(cloud->at(i), max_search_radius, k_indices, k_sqr_distances, kNearestNeighbors) <= 0) {
      continue;
    }

    double delta_p = 0;
    double sum_w   = 0;
    auto p         = cloud->points[i].getVector3fMap();
    for (int j : k_indices) {
      auto q     = cloud->points[j].getVector3fMap();
      double d_d = (q - p).norm();
      double d_n = (q - p).dot(normals[i]);
      double w   = std::exp(-d_d * d_d / (2 * sigma_d * sigma_d) - d_n * d_n / (2 * sigma_n * sigma_n));
      delta_p += w * d_n;
      sum_w += w;
    }
    p += delta_p / sum_w * normals[i];
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = true;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 6, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<Eigen::Vector3f> centers;
  std::vector<Eigen::Vector3f> normals;

  LoadFullPointCloud(FLAGS_project_input_path, cloud, centers);
  PcaEstimateNormal(cloud, centers, kNearestNeighbors, kDownsampleVoxelSize, normals);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  DownsamplePointCloud(cloud, cloud_downsampled, kDownsampleVoxelSize);

  DLOG(INFO) << "Smoothing...";
  pcl::io::savePCDFileBinary(FLAGS_project_output_path + "/before-smooth.pcd", *cloud);
  SmoothPointCloud(normals, kSmoothMaxNearestNeighbors, kSmoothMaxSearchRadius, kSmoothSigmaD, kSmoothSigmaN, cloud);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud->size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud->points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    np.intensity              = cloud->points[i].intensity;
    cloud_with_normals->points.push_back(np);
  }

  DLOG(INFO) << "Saving cloud with normals...";
  pcl::io::savePCDFileBinary(FLAGS_project_output_path + "/normals.pcd", *cloud_with_normals);
  DLOG(INFO) << "Save to " << FLAGS_project_output_path + "/normals.pcd";

  return 0;
}
