
#include <glog/logging.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/filters/fast_bilateral.h>
#include <pcl/filters/fast_bilateral_omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <fstream>
#include <pcl/filters/impl/fast_bilateral.hpp>
#include <pcl/filters/impl/fast_bilateral_omp.hpp>
#include <thread>
#include <vector>

#include "map/utils.h"

DEFINE_string(las_filename, "e:/BaiduNetdiskDownload/colorized.las", "Input project path");
DEFINE_bool(output_full, true, "Output full point cloud");

static constexpr double kDownsampleVoxelSize    = 0.05;
static constexpr int kNearestNeighbors          = 15;
static constexpr int kSmoothMaxNearestNeighbors = 100;
static constexpr double kSmoothMaxSearchRadius  = 0.3;
static constexpr double kSmoothSigmaD           = 0.05;
static constexpr double kSmoothSigmaN           = 0.05;

#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/LasReader.hpp>

void LoadLAS(const std::string &filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
  // 初始化点云对象
  cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

  // 创建 PDAL 读取器
  pdal::StageFactory factory;
  pdal::Stage *reader = factory.createStage("readers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", filename));
  reader->setOptions(opts);

  // 准备点云数据容器
  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  // 配置点云属性
  cloud->width    = view->size();
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width);

  // 遍历并转换数据
  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZI p;
    p.x              = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    p.y              = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    p.z              = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.intensity      = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);
    cloud->points[i] = p;
  }
}

void SavePointCloud(const std::string &filename, const pcl::PointCloud<pcl::PointXYZI> &cloud, const std::vector<Eigen::Vector3f> &normals) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud.size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud.points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    cloud_out->points.push_back(np);
  }
}

void PcaEstimateNormalNoDirect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
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
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<Eigen::Vector3f> normals;

  CHECK(boost::filesystem::is_regular_file(FLAGS_las_filename));

  DLOG(INFO) << "Loading LAS file...";
  LoadLAS(FLAGS_las_filename, cloud);
  DLOG(INFO) << "Loaded " << cloud->size() << " points.";

  if (!FLAGS_output_full) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    DownsamplePointCloud(cloud, cloud_downsampled, kDownsampleVoxelSize);
    cloud = cloud_downsampled;
  }

  DLOG(INFO) << "Saving original cloud...";
  PcaEstimateNormalNoDirect(cloud, kNearestNeighbors, kDownsampleVoxelSize, normals);

  DLOG(INFO) << "Smoothing...";
  // pcl::io::savePCDFileBinary(FLAGS_project_output_path + "/before-smooth.pcd", *cloud);
  SmoothPointCloud(normals, kSmoothMaxNearestNeighbors, kSmoothMaxSearchRadius, kSmoothSigmaD, kSmoothSigmaN, cloud);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud->size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud->points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    np.intensity              = cloud->points[i].intensity;
    cloud_with_normals->push_back(np);
  }

  DLOG(INFO) << "Saving cloud with normals...";
  pcl::io::savePCDFileBinary(FLAGS_las_filename + "_normals.pcd", *cloud_with_normals);
  DLOG(INFO) << "Save to " << FLAGS_las_filename + "_normals.pcd";

  std::cout << "done." << std::endl;

  return 0;
}
