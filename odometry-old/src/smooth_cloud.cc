#include <glog/logging.h>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <thread>
#include <vector>

DEFINE_string(input_pcd, "/path/to/scans.pcd", "Input PCD file path");
DEFINE_string(output_pcd, "/path/to/output_smoothed.pcd", "Output smoothed PCD file path");
DEFINE_double(voxel_size, 0.05, "Downsample voxel size (meters)");

static constexpr int kSmoothMaxNearestNeighbors = 100;
static constexpr double kSmoothMaxSearchRadius  = 0.3;
static constexpr double kSmoothSigmaD           = 0.05;
static constexpr double kSmoothSigmaN           = 0.05;

// ------------------------------------------------------------
// 下采样函数
// ------------------------------------------------------------
void DownsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &output,
                          double voxel_size) {
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(input);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*output);
}

// ------------------------------------------------------------
// 基于 PCA 估计法线（用下采样点云建树）
// ------------------------------------------------------------
void EstimateNormals(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                     const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud_downsampled,
                     int k,
                     std::vector<Eigen::Vector3f> &normals) {
  normals.resize(cloud->size());
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
  tree->setInputCloud(cloud_downsampled);

  DLOG(INFO) << "Estimating normals using downsampled KD-tree...";

#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(k);
    std::vector<float> k_sqr_distances(k);
    if (tree->nearestKSearch(cloud->at(i), k, k_indices, k_sqr_distances) <= 3)
      continue;

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
    normals[i] = eigensolver.eigenvectors().col(0).cast<float>();
  }
}

// ------------------------------------------------------------
// 平滑函数（沿法线方向滤波）
// ------------------------------------------------------------
void SmoothPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                      const std::vector<Eigen::Vector3f> &normals,
                      int kNearestNeighbors,
                      double max_search_radius,
                      double sigma_d,
                      double sigma_n) {
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(cloud);

  DLOG(INFO) << "Smoothing points...";
#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(kNearestNeighbors);
    std::vector<float> k_sqr_distances(kNearestNeighbors);

    if (tree->radiusSearch(cloud->at(i), max_search_radius, k_indices, k_sqr_distances, kNearestNeighbors) <= 0)
      continue;

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

    if (sum_w > 0)
      p += delta_p / sum_w * normals[i];
  }
}

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  FLAGS_logtostderr = true;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  if (pcl::io::loadPCDFile(FLAGS_input_pcd, *cloud) < 0) {
    DLOG(ERROR) << "Failed to load PCD: " << FLAGS_input_pcd;
    return -1;
  }
  DLOG(INFO) << "Loaded " << cloud->size() << " points.";

  // Step 1: 下采样
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  DownsamplePointCloud(cloud, cloud_downsampled, FLAGS_voxel_size);
  DLOG(INFO) << "Downsampled to " << cloud_downsampled->size() << " points.";

  // Step 2: 基于下采样点云构建 KDTree 估计法线
  std::vector<Eigen::Vector3f> normals;
  EstimateNormals(cloud, cloud_downsampled, 15, normals);

  // Step 3: 平滑
  SmoothPointCloud(cloud, normals, kSmoothMaxNearestNeighbors,
                   kSmoothMaxSearchRadius, kSmoothSigmaD, kSmoothSigmaN);

  // Step 4: 保存
  pcl::io::savePCDFileBinary(FLAGS_output_pcd, *cloud);
  DLOG(INFO) << "Saved smoothed cloud to " << FLAGS_output_pcd;

  std::cout << "done." << std::endl;
  return 0;
}
