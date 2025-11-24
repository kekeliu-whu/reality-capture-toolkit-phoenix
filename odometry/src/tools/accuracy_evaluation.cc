#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/LasReader.hpp>

#include <ceres/ceres.h>
#include <crashpad/client/crash_report_database.h>
#include <crashpad/client/crashpad_client.h>
#include <crashpad/client/settings.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/common/transforms.h>
#include <pcl/filters/statistical_outlier_removal.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/segmentation/extract_clusters.h>
#include <boost/filesystem.hpp>

DEFINE_string(las_filename, "D:/Users/rick/Desktop/slam_evaluation/l2pro/result-process/eval.las", "");
DEFINE_string(gt_filename, "D:/Users/rick/Desktop/slam_evaluation/l2pro/result-process/gt.las", "");

void LoadLAS(const std::string& filename, pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud) {
  cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

  pdal::StageFactory factory;
  pdal::Stage* reader = factory.createStage("readers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", filename));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  cloud->height   = 1;
  cloud->is_dense = true;

  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZI p;
    p.x         = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    p.y         = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    p.z         = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.intensity = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);
    cloud->push_back(p);
  }
}

// 欧几里得聚类提取目标
void ExtractClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& input, std::vector<pcl::PointIndices>& cluster_indices) {
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(input);

  pcl::EuclideanClusterExtraction<pcl::PointXYZI> ec;
  ec.setClusterTolerance(0.03);  // 3cm
  ec.setMinClusterSize(200);
  ec.setMaxClusterSize(5000);
  ec.setSearchMethod(tree);
  ec.setInputCloud(input);
  ec.extract(cluster_indices);
}

// 提取有效聚类并保存聚类点和中心
pcl::PointCloud<pcl::PointXYZI>::Ptr ExtractAndSaveValidClusters(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cloud_filtered,
                                                                 const std::vector<pcl::PointIndices>& cluster_indices,
                                                                 const std::string& filename_prefix) {
  pcl::PointCloud<pcl::PointXYZI>::Ptr all_clusters(new pcl::PointCloud<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_centers(new pcl::PointCloud<pcl::PointXYZI>);
  int j = 0;
  for (const auto& indices : cluster_indices) {
    // 计算聚类的最大半径
    float max_radius = 0.0f;
    Eigen::Vector4f centroid(0, 0, 0, 0);
    if (!indices.indices.empty()) {
      pcl::compute3DCentroid(*cloud_filtered, indices.indices, centroid);
      for (int idx : indices.indices) {
        const auto& pt = (*cloud_filtered)[idx];
        float dist     = std::sqrt((pt.x - centroid[0]) * (pt.x - centroid[0]) + (pt.y - centroid[1]) * (pt.y - centroid[1]) +
                                   (pt.z - centroid[2]) * (pt.z - centroid[2]));
        if (dist > max_radius) max_radius = dist;
      }
    }
    if (max_radius <= 0.20f) {  // 只保留半径<=10cm的聚类
      for (int idx : indices.indices) {
        pcl::PointXYZI pt = (*cloud_filtered)[idx];
        pt.intensity      = static_cast<float>(j);  // 用聚类编号区分
        all_clusters->push_back(pt);
      }
      pcl::PointXYZI center_pt;
      center_pt.x         = centroid[0];
      center_pt.y         = centroid[1];
      center_pt.z         = centroid[2];
      center_pt.intensity = static_cast<float>(j);
      cluster_centers->push_back(center_pt);
      j++;
    }
  }
  DLOG(INFO) << "Valid clusters size: " << j;
  all_clusters->width    = all_clusters->size();
  all_clusters->height   = 1;
  all_clusters->is_dense = true;
  pcl::io::savePCDFileBinary(filename_prefix + "_clusters.pcd", *all_clusters);

  cluster_centers->width    = cluster_centers->size();
  cluster_centers->height   = 1;
  cluster_centers->is_dense = true;
  pcl::io::savePCDFileBinary(filename_prefix + "_cluster_centers.pcd", *cluster_centers);

  return cluster_centers;
}

// 定义点对之间的残差函数（即误差函数）
struct PointMatchingCostFunctor {
  PointMatchingCostFunctor(const Eigen::Vector3d& source_point, const Eigen::Vector3d& target_point)
      : source_point_(source_point), target_point_(target_point) {}

  // 实现残差计算
  template <typename T>
  bool operator()(const T* const quaternion, const T* const translation, T* residual) const {
    // 将四元数表示的旋转转换为旋转矩阵
    Eigen::Quaternion<T> q(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    q.normalize();

    // 对源点进行旋转和平移变换
    Eigen::Matrix<T, 3, 1> transformed_point = q * source_point_.cast<T>() + Eigen::Matrix<T, 3, 1>(translation[0], translation[1], translation[2]);

    // 计算变换后的源点与目标点之间的残差（欧氏距离）
    residual[0] = transformed_point[0] - T(target_point_[0]);
    residual[1] = transformed_point[1] - T(target_point_[1]);
    residual[2] = transformed_point[2] - T(target_point_[2]);

    return true;
  }

  // 源点和目标点
  const Eigen::Vector3d source_point_;
  const Eigen::Vector3d target_point_;
};

// Evaluate distance between cluster centers and ground truth
void PerformEvaluation(const pcl::PointCloud<pcl::PointXYZI>::Ptr& cluster_centers, const pcl::PointCloud<pcl::PointXYZI>::Ptr& gt_points) {
  pcl::io::savePCDFileBinary("cluster_centers_dgb.pcd", *cluster_centers);
  pcl::io::savePCDFileBinary("gt_points_dbg.pcd", *gt_points);

  DLOG(INFO) << "cluster size: " << cluster_centers->size();
  DLOG(INFO) << "ground truth size: " << gt_points->size();

  // Step 1: Use KD-tree to establish nearest neighbor correspondences
  pcl::KdTreeFLANN<pcl::PointXYZI> kdtree;
  kdtree.setInputCloud(gt_points);

  for (auto& e : *gt_points) {
    LOG(INFO) << "GT Point: " << e.x << " " << e.y << " " << e.z;
  }

  // Initialize optimization variables: rotation matrix (quaternion) and translation vector
  // These will be shared across iterations
  double quaternion[4]  = {1.0, 0.0, 0.0, 0.0};  // Initial identity quaternion
  double translation[3] = {0.0, 0.0, 0.0};       // Initial zero translation

  // Make a copy of the original cluster centers for iterative refinement
  pcl::PointCloud<pcl::PointXYZI>::Ptr current_clusters(new pcl::PointCloud<pcl::PointXYZI>());
  *current_clusters = *cluster_centers;

  // Define distance thresholds for multiple iterations
  std::vector<float> distance_thresholds = {1.0, 0.8, 0.4};  // First 1m, then 0.4m
  int num_iterations                     = distance_thresholds.size();

  // Store best transformation for final evaluation
  Eigen::Matrix4f best_transform = Eigen::Matrix4f::Identity();
  std::vector<std::pair<int, int>> final_correspondences;

  // Perform multiple iterations of point matching and optimization
  for (int iter = 0; iter < num_iterations; ++iter) {
    float max_distance = distance_thresholds[iter];
    DLOG(INFO) << "Iteration " << iter + 1 << " with max distance: " << max_distance << " m";

    // Save corresponding point pairs for this iteration
    std::vector<std::pair<int, int>> correspondences;

    // Find nearest GT point for each cluster center with current distance threshold
    for (int i = 0; i < current_clusters->size(); ++i) {
      std::vector<int> indices(1);
      std::vector<float> distances(1);

      if (kdtree.nearestKSearch(current_clusters->points[i], 1, indices, distances) > 0) {
        LOG(INFO) << "Cluster Point: " << current_clusters->points[i].x << " "
                  << current_clusters->points[i].y << " " << current_clusters->points[i].z
                  << " -> Nearest GT Point: " << gt_points->points[indices[0]].x << " "
                  << gt_points->points[indices[0]].y << " " << gt_points->points[indices[0]].z << ", Distance: " << sqrt(distances[0]);
        if (distances[0] <= max_distance * max_distance) {
          // Record correspondence
          correspondences.push_back({i, indices[0]});
        }
      }
    }

    DLOG(INFO) << "Iteration " << iter + 1 << " found " << correspondences.size() << " correspondences";

    if (correspondences.size() < 3) {
      DLOG(ERROR) << "Not enough correspondences in iteration " << iter + 1;

      // If it's the first iteration and not enough correspondences, exit
      if (iter == 0) {
        return;
      }

      // Otherwise, use previous iteration results and continue
      continue;
    }

    // Create Ceres optimization problem for this iteration
    ceres::Problem problem;

    // Add residual blocks for each corresponding point pair
    for (const auto& corr : correspondences) {
      Eigen::Vector3d source_point(current_clusters->points[corr.first].x, current_clusters->points[corr.first].y,
                                   current_clusters->points[corr.first].z);

      Eigen::Vector3d target_point(gt_points->points[corr.second].x, gt_points->points[corr.second].y, gt_points->points[corr.second].z);

      ceres::CostFunction* cost_function =
          new ceres::AutoDiffCostFunction<PointMatchingCostFunctor, 3, 4, 3>(new PointMatchingCostFunctor(source_point, target_point));

      problem.AddResidualBlock(cost_function, new ceres::HuberLoss(0.2), quaternion, translation);
    }

    // Add quaternion manifold to ensure unit quaternion
    ceres::Manifold* quaternion_parameterization = new ceres::QuaternionManifold();
    problem.SetManifold(quaternion, quaternion_parameterization);

    // Set solver options
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = 100;

    // Run optimization
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);

    DLOG(INFO) << "Iteration " << iter + 1 << " Ceres optimization completed: " << summary.BriefReport();

    // Build transformation matrix from optimization results
    Eigen::Quaterniond q(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    q.normalize();
    Eigen::Matrix3d rotation_matrix = q.toRotationMatrix();

    // Build 4x4 transformation matrix
    Eigen::Matrix4f transform   = Eigen::Matrix4f::Identity();
    transform.block<3, 3>(0, 0) = rotation_matrix.cast<float>();
    transform.block<3, 1>(0, 3) = Eigen::Vector3f(translation[0], translation[1], translation[2]);

    // Apply transformation
    pcl::PointCloud<pcl::PointXYZI>::Ptr transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
    pcl::transformPointCloud(*cluster_centers, *transformed_cloud, transform);

    // Save the current transformation for the next iteration
    *current_clusters = *transformed_cloud;
    best_transform    = transform;

    // Save the final iteration's correspondences for error evaluation
    if (iter == num_iterations - 1) {
      final_correspondences = correspondences;
    }

    // Save intermediate transformed cloud for debugging
    std::string filename = "transformed_iter_" + std::to_string(iter + 1) + ".pcd";
    pcl::io::savePCDFileBinary(filename, *transformed_cloud);
    DLOG(INFO) << "Saved " << filename;
  }

  // Apply the final best transformation
  pcl::PointCloud<pcl::PointXYZI>::Ptr final_transformed_cloud(new pcl::PointCloud<pcl::PointXYZI>());
  pcl::transformPointCloud(*cluster_centers, *final_transformed_cloud, best_transform);

  // Calculate and evaluate errors
  float sum_error = 0.0f;
  float max_error = 0.0f;
  std::vector<float> errors;

  // Horizontal and vertical error metrics
  float sum_horizontal_error = 0.0f;
  float max_horizontal_error = 0.0f;
  float sum_vertical_error   = 0.0f;
  float max_vertical_error   = 0.0f;
  std::vector<float> horizontal_errors;
  std::vector<float> vertical_errors;

  for (const auto& corr : final_correspondences) {
    Eigen::Vector3f pt_src(final_transformed_cloud->points[corr.first].x, final_transformed_cloud->points[corr.first].y,
                           final_transformed_cloud->points[corr.first].z);

    Eigen::Vector3f pt_tgt(gt_points->points[corr.second].x, gt_points->points[corr.second].y, gt_points->points[corr.second].z);

    // Total 3D error
    float error = (pt_src - pt_tgt).norm();
    sum_error += error;
    max_error = std::max(max_error, error);
    errors.push_back(error);

    // Horizontal error (XY plane)
    float horizontal_error = std::sqrt((pt_src.x() - pt_tgt.x()) * (pt_src.x() - pt_tgt.x()) + (pt_src.y() - pt_tgt.y()) * (pt_src.y() - pt_tgt.y()));
    sum_horizontal_error += horizontal_error;
    max_horizontal_error = std::max(max_horizontal_error, horizontal_error);
    horizontal_errors.push_back(horizontal_error);

    // Vertical error (Z axis)
    float vertical_error = std::abs(pt_src.z() - pt_tgt.z());
    sum_vertical_error += vertical_error;
    max_vertical_error = std::max(max_vertical_error, vertical_error);
    vertical_errors.push_back(vertical_error);

    // Log detailed error information for each point
    LOG(INFO) << "Point " << gt_points->points[corr.second].intensity << " - Error: " << error << " m, Horizontal: " << horizontal_error
              << " m, Vertical: " << vertical_error
              << " m";
  }

  // Calculate mean errors
  float mean_error            = sum_error / static_cast<float>(final_correspondences.size());
  float mean_horizontal_error = sum_horizontal_error / static_cast<float>(final_correspondences.size());
  float mean_vertical_error   = sum_vertical_error / static_cast<float>(final_correspondences.size());

  // Calculate RMSE
  float squared_sum = 0.0f;
  for (const auto& err : errors) {
    squared_sum += err * err;
  }
  float rmse = std::sqrt(squared_sum / static_cast<float>(errors.size()));

  // Calculate standard deviation
  float variance = 0.0f;
  for (const auto& err : errors) {
    variance += (err - mean_error) * (err - mean_error);
  }
  float std_dev = std::sqrt(variance / static_cast<float>(errors.size()));

  // Output evaluation results
  DLOG(INFO) << "============ Results ============";
  DLOG(INFO) << "Point correspondence count: " << final_correspondences.size();
  DLOG(INFO) << "3D RMSE: " << rmse << " m";
  DLOG(INFO) << "3D average error: " << mean_error << " m";
  DLOG(INFO) << "3D max error: " << max_error << " m";
  DLOG(INFO) << "3D std error: " << std_dev << " m";

  // Output horizontal and vertical error metrics
  DLOG(INFO) << "Horizontal average error: " << mean_horizontal_error << " m";
  DLOG(INFO) << "Horizontal max error: " << max_horizontal_error << " m";
  DLOG(INFO) << "Vertical average error: " << mean_vertical_error << " m";
  DLOG(INFO) << "Vertical max error: " << max_vertical_error << " m";

  // Output transformation matrix
  DLOG(INFO) << "Best rotation matrix R:";
  Eigen::Matrix3f final_rotation = best_transform.block<3, 3>(0, 0);
  for (int i = 0; i < 3; ++i) {
    DLOG(INFO) << final_rotation(i, 0) << " " << final_rotation(i, 1) << " " << final_rotation(i, 2);
  }

  DLOG(INFO) << "Best translation vector t:";
  Eigen::Vector3f final_translation = best_transform.block<3, 1>(0, 3);
  DLOG(INFO) << final_translation.x() << " " << final_translation.y() << " " << final_translation.z();

  // Save the final transformed point cloud
  pcl::io::savePCDFileBinary("transformed_clusters.pcd", *final_transformed_cloud);
  DLOG(INFO) << "Saved transformed_clusters.pcd";
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  /////////////////////////////////// setup glog ///////////////////////////////////
  google::InitGoogleLogging(argv[0]);
  // EncryptedLogSink *sink = new EncryptedLogSink();
  // google::AddLogSink(sink);
  FLAGS_logtostderr = 1;

  std::shared_ptr<void> done{nullptr, [](void*) {
                               DLOG(INFO) << "Shutdown glog.";
                               google::ShutdownGoogleLogging();
                             }};

  /////////////////////////////////// setup omp ///////////////////////////////////
  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  /////////////////////////////////// setup crashpad /////////////////////////////
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_filtered;
  DCHECK(boost::filesystem::is_regular_file(FLAGS_las_filename));
  LoadLAS(FLAGS_las_filename, cloud_filtered);
  DLOG(INFO) << "Cloud size filtered: " << cloud_filtered->size();
  pcl::io::savePCDFileBinary(FLAGS_las_filename + ".pcd", *cloud_filtered);

  // ============================================
  // 欧几里得聚类提取目标
  std::vector<pcl::PointIndices> cluster_indices;
  ExtractClusters(cloud_filtered, cluster_indices);

  DLOG(INFO) << "Detected " << cluster_indices.size() << " clusters.";

  // 提取有效聚类并保存，并获取聚类中心
  pcl::PointCloud<pcl::PointXYZI>::Ptr cluster_centers = ExtractAndSaveValidClusters(cloud_filtered, cluster_indices, FLAGS_las_filename);

  pcl::PointCloud<pcl::PointXYZI>::Ptr tag_groundtruth(new pcl::PointCloud<pcl::PointXYZI>);
  LoadLAS(FLAGS_gt_filename, tag_groundtruth);
  PerformEvaluation(cluster_centers, tag_groundtruth);

  return 0;
}