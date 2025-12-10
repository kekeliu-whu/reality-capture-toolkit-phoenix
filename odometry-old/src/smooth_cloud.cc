#include <gflags/gflags.h>
#include <omp.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <spdlog/spdlog.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <thread>
#include <vector>

#include "migration/inc_las_writer.h"
#include "migration/logging.h"
#include "migration/string.h"

DEFINE_string(input_las, "C:\\4.indoor-big-slow\\hall\\out\\map_aligned.las", "Input LAS file path");
DEFINE_string(output_las, "C:\\4.indoor-big-slow\\hall\\out\\map_smooth.las", "Output smoothed LAS file path");

static constexpr int    kSmoothMaxNearestNeighbors = 80;
static constexpr double kSmoothMaxSearchRadius     = 0.3;
static constexpr double kSmoothSigmaD              = 0.05;
static constexpr double kSmoothSigmaN              = 0.05;
static constexpr double kVoxelSize                 = 0.05;

// Intensity statistics structure
struct IntensityStats {
  double              min_intensity;
  double              max_intensity;
  double              mean_intensity;
  double              std_dev_intensity;
  std::vector<double> histogram;
  std::vector<double> cdf;
};

// Analyze intensity values in point cloud
IntensityStats AnalyzeIntensity(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud) {
  IntensityStats stats;

  if (cloud->empty()) {
    spdlog::error("Point cloud is empty, cannot analyze intensity.");
    return stats;
  }

  // 1. collect intensity values
  std::vector<double> intensities;
  intensities.reserve(cloud->size());

  for (const auto &point : cloud->points) {
    intensities.push_back(static_cast<double>(point.intensity));
  }

  // 2. calculate basic statistics
  stats.min_intensity = *std::min_element(intensities.begin(), intensities.end());
  stats.max_intensity = *std::max_element(intensities.begin(), intensities.end());

  double sum           = std::accumulate(intensities.begin(), intensities.end(), 0.0);
  stats.mean_intensity = sum / intensities.size();

  double sq_sum           = std::inner_product(intensities.begin(), intensities.end(), intensities.begin(), 0.0);
  stats.std_dev_intensity = std::sqrt(sq_sum / intensities.size() - stats.mean_intensity * stats.mean_intensity);

  spdlog::info("Analyzed {} points for intensity statistics.", cloud->size());
  spdlog::info("Intensity range: [{}, {}], mean: {}, std_dev: {}", stats.min_intensity, stats.max_intensity,
               stats.mean_intensity, stats.std_dev_intensity);

  // 3. Compute histogram
  const int num_bins = 256;
  stats.histogram.resize(num_bins, 0.0);

  for (double intensity : intensities) {
    int bin = static_cast<int>((intensity - stats.min_intensity) / (stats.max_intensity - stats.min_intensity) *
                               (num_bins - 1));
    bin     = std::max(0, std::min(bin, num_bins - 1));
    stats.histogram[bin] += 1.0;
  }

  // 4. Compute CDF
  stats.cdf.resize(num_bins, 0.0);
  stats.cdf[0] = stats.histogram[0];

  for (int i = 1; i < num_bins; ++i) {
    stats.cdf[i] = stats.cdf[i - 1] + stats.histogram[i];
  }

  // Normalize CDF
  for (int i = 0; i < num_bins; ++i) {
    stats.cdf[i] /= cloud->size();
  }

  return stats;
}

// Stretch intensity using histogram equalization (CDF-based)
void StretchIntensityHistogramEqualization(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const IntensityStats &stats,
                                           double target_max = 255.0) {
  if (stats.max_intensity == stats.min_intensity) {
    spdlog::warn("Intensity range is zero, skipping histogram equalization");
    return;
  }

  const int num_bins = 256;

#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    double intensity = cloud->points[i].intensity;
    int    bin = static_cast<int>((intensity - stats.min_intensity) / (stats.max_intensity - stats.min_intensity) *
                                  (num_bins - 1));
    bin        = std::max(0, std::min(bin, num_bins - 1));

    // Map using CDF
    cloud->points[i].intensity = stats.cdf[bin] * target_max;
  }

  spdlog::info("Histogram equalization applied");
}

void ReadLidarPoints(const std::string &filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                     std::vector<double> &timestamps) {
  pdal::StageFactory factory;
  pdal::Stage       *reader = factory.createStage("readers.las");
  pdal::Options      opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  cloud.reset(new pcl::PointCloud<pcl::PointXYZI>());
  cloud->reserve(view->size());
  timestamps.clear();
  timestamps.reserve(view->size());
  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZI p;
    p.x         = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    p.y         = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    p.z         = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.intensity = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);

    cloud->push_back(p);
    timestamps.push_back(view->getFieldAs<double>(pdal::Dimension::Id::GpsTime, i));
  }

  spdlog::info("Loaded {} lidar points from {}", cloud->size(), filename);
}

// ------------------------------------------------------------
// Downsample point cloud using voxel grid filter
// ------------------------------------------------------------
void DownsamplePointCloud(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &input,
                          pcl::PointCloud<pcl::PointXYZI>::Ptr &output, double voxel_size) {
  pcl::VoxelGrid<pcl::PointXYZI> sor;
  sor.setInputCloud(input);
  sor.setLeafSize(voxel_size, voxel_size, voxel_size);
  sor.filter(*output);
}

// ------------------------------------------------------------
// Estimate normals for each point in the cloud using k-nearest neighbors
// ------------------------------------------------------------
void EstimateNormals(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud,
                     const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud_downsampled, int k,
                     std::vector<Eigen::Vector3f> &normals) {
  normals.resize(cloud->size());
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
  tree->setInputCloud(cloud_downsampled);

  spdlog::info("Estimating normals using downsampled KD-tree...");

#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int>   k_indices(k);
    std::vector<float> k_sqr_distances(k);
    if (tree->nearestKSearch(cloud->at(i), k, k_indices, k_sqr_distances) <= 3) continue;

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
// Smooth point cloud using bilateral-like filtering
// ------------------------------------------------------------
void SmoothPointCloud(pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, const std::vector<Eigen::Vector3f> &normals,
                      int kNearestNeighbors, double max_search_radius, double sigma_d, double sigma_n) {
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(cloud);

  spdlog::info("Smoothing points...");
#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int>   k_indices(kNearestNeighbors);
    std::vector<float> k_sqr_distances(kNearestNeighbors);

    if (tree->radiusSearch(cloud->at(i), max_search_radius, k_indices, k_sqr_distances, kNearestNeighbors) <= 0)
      continue;

    double delta_p = 0;
    double sum_w   = 0;
    auto   p       = cloud->points[i].getVector3fMap();

    for (int j : k_indices) {
      auto   q   = cloud->points[j].getVector3fMap();
      double d_d = (q - p).norm();
      double d_n = (q - p).dot(normals[i]);
      double w   = std::exp(-d_d * d_d / (2 * sigma_d * sigma_d) - d_n * d_n / (2 * sigma_n * sigma_n));
      delta_p += w * d_n;
      sum_w += w;
    }

    if (sum_w > 0) p += delta_p / sum_w * normals[i];
  }
}

void WriteLasFile(const std::string &filename, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud,
                  const std::vector<double> &timestamps) {
  spdlog::info("Starting incremental LAS write example...");
  // 1. Define point cloud data layout
  pdal::PointTable table;
  table.layout()->registerDim(pdal::Dimension::Id::GpsTime);
  table.layout()->registerDim(pdal::Dimension::Id::X);
  table.layout()->registerDim(pdal::Dimension::Id::Y);
  table.layout()->registerDim(pdal::Dimension::Id::Z);
  table.layout()->registerDim(pdal::Dimension::Id::Intensity);

  // 2. Initialize writer
  migration::IncrementalLasWriter writer;
  writer.initialize(filename, table);
  spdlog::info("Writer initialized for file: output_huge.las");

  // 3. Create PointView and fill with data
  pdal::PointViewPtr view(new pdal::PointView(table));
  for (int i = 0; i < cloud->size(); ++i) {
    view->setField(pdal::Dimension::Id::X, i, cloud->points[i].x);
    view->setField(pdal::Dimension::Id::Y, i, cloud->points[i].y);
    view->setField(pdal::Dimension::Id::Z, i, cloud->points[i].z);
    view->setField(pdal::Dimension::Id::Intensity, i, (uint16_t)cloud->points[i].intensity);
    view->setField(pdal::Dimension::Id::GpsTime, i, timestamps[i]);
  }
  writer.writeView(view);

  // 4. Finalize - execute write operation once
  writer.finalize(table);

  spdlog::info("Write operation completed successfully!");
}

// ------------------------------------------------------------
// main()
// ------------------------------------------------------------
int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  InitSpdLog();

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);
  spdlog::info("Using {} / {} cores.", cores_used, cores);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<double>                  timestamps;
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<Eigen::Vector3f>         normals;

  // Step 0: Read input point cloud
  ReadLidarPoints(FLAGS_input_las, cloud, timestamps);

  // Step 0.5: Analyze intensity
  IntensityStats intensity_stats = AnalyzeIntensity(cloud);

  StretchIntensityHistogramEqualization(cloud, intensity_stats, 255.0);

  // Step 1: Downsample point cloud using voxel grid filter
  DownsamplePointCloud(cloud, cloud_downsampled, kVoxelSize);
  spdlog::info("Downsampled to {} points.", cloud_downsampled->size());

  // Step 2: Estimate normals for each point in the cloud using k-nearest neighbors
  EstimateNormals(cloud, cloud_downsampled, 15, normals);

  // Step 3: Smooth point cloud using bilateral-like filtering
  SmoothPointCloud(cloud, normals, kSmoothMaxNearestNeighbors, kSmoothMaxSearchRadius, kSmoothSigmaD, kSmoothSigmaN);

  // Step 4: Save smoothed point cloud
  WriteLasFile(FLAGS_output_las, cloud, timestamps);

  spdlog::info("done.");
  return 0;
}
