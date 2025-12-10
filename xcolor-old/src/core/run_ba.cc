
#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/geometry/triangulation.h>
#include <colmap/scene/database.h>
#include <colmap/scene/reconstruction.h>
#include <spdlog/spdlog.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>

#include "common/histogram.h"
#include "core/run_ba.h"
#include "core/xsfm_lib.h"
#include "io/colmap_io.h"
#include "io/io_utils.h"

namespace {

xcolor::TrackConstraintType TryTriangulate(const xcolor::SfmConfig &config, const colmap::Image &image1, const colmap::Image &image2,
                                           const colmap::Camera &camera1, const colmap::Camera &camera2, const xcolor::Point2DInfo &point_on_image1,
                                           const xcolor::Point2DInfo &point_in_image2, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                                           const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, int iter, Eigen::Vector3d &point3D,
                                           Eigen::Vector3d &lidar_point, Eigen::Vector3d &lidar_normal) {
  auto cam_from_world1 = image1.CamFromWorld().ToMatrix();
  auto cam_from_world2 = image2.CamFromWorld().ToMatrix();

  // if colmap::TriangulatePoint failed, ignore the match
  bool ok = colmap::TriangulatePoint(cam_from_world1, cam_from_world2, camera1.CamFromImg(point_on_image1.point_pixel),
                                     camera2.CamFromImg(point_in_image2.point_pixel), &point3D);
  if (!ok) {
    return xcolor::TrackConstraintType::kUnconstrained;
  }

  Eigen::Vector3d cam1_center = -(cam_from_world1.block<3, 3>(0, 0).transpose() * cam_from_world1.block<3, 1>(0, 3));
  Eigen::Vector3d cam2_center = -(cam_from_world2.block<3, 3>(0, 0).transpose() * cam_from_world2.block<3, 1>(0, 3));

  // if the viewing angle between the two perspectives is too small, the match will be ignored
  double angle = colmap::CalculateTriangulationAngle(cam1_center, cam2_center, point3D);
  if (std::abs(angle) < config.min_tri_angle * M_PI / 180) {
    return xcolor::TrackConstraintType::kUnconstrained;
  }

  // if reprojection error is too large, the match will be ignored
  auto projection_error1 = xcolor::ComputePixelError(point3D, point_on_image1.point_pixel, image1, camera1);
  auto projection_error2 = xcolor::ComputePixelError(point3D, point_in_image2.point_pixel, image2, camera2);
  if (projection_error1.norm() > config.reproject_error_outlier_thresholds_twoview[iter] ||
      projection_error2.norm() > config.reproject_error_outlier_thresholds_twoview[iter]) {
    return xcolor::TrackConstraintType::kUnconstrained;
  }

  std::vector<float> distances;
  std::vector<int> indices;
  kdtree.nearestKSearch(pcl::PointXYZINormal{(float)point3D.x(), (float)point3D.y(), (float)point3D.z()}, 1, indices, distances);

  auto nearest_point = point_cloud[indices[0]];
  lidar_point        = nearest_point.getVector3fMap().cast<double>();
  lidar_normal       = nearest_point.getNormalVector3fMap().cast<double>();

  // if lidar error is too large, the match will be ignored
  auto lidar_error = xcolor::ComputeLidarError(point3D, lidar_point, lidar_normal);
  if (std::abs(lidar_error) > config.lidar_error_outlier_thresholds_twoview[iter]) {
    return xcolor::TrackConstraintType::kVisualOnly;
  }

  return xcolor::TrackConstraintType::kVisualAndLidar;
}

}  // namespace

namespace xcolor {

void RunTwoViewBA(const xcolor::SfmConfig &config, const std::string &output_path, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                  const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                  std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                  std::vector<xcolor::MatchTrack> &match_tracks) {
  for (int iter = 0; iter < config.outer_opt_num_two_view; ++iter) {
    ceres::Problem problem;
    auto loss_function_image = new ceres::CauchyLoss(config.reproject_cauchy_loss_scale);
    auto loss_function_lidar =
        new ceres::ScaledLoss(new ceres::CauchyLoss(config.lidar_cauchy_weight), config.lidar_weight_scale, ceres::TAKE_OWNERSHIP);
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    std::vector<ceres::ResidualBlockId> lidar_residual_block_ids;

    std::unordered_set<colmap::image_t> optimized_image_ids;

    int count = 0;
#pragma omp parallel for
    for (int i = 0; i < match_tracks.size(); ++i) {
      auto &match_pair      = match_tracks[i];
      auto &point_on_image1 = *match_pair.point2D_on_imageN[0];
      auto &point_in_image2 = *match_pair.point2D_on_imageN[1];
      auto &image1          = images[point_on_image1.image_id];
      auto &image2          = images[point_in_image2.image_id];
      auto &camera1         = cameras[point_on_image1.camera_id];
      auto &camera2         = cameras[point_in_image2.camera_id];
      auto &point3D         = match_tracks[i].point3D.point3D;
      auto &constraint_type = match_tracks[i].constraint_type;
      Eigen::Vector3d lidar_point;
      Eigen::Vector3d lidar_normal;

      if (!image1.HasPose()) { spdlog::error("Image {} has no pose.", point_on_image1.image_id); exit(1); }
      if (!image2.HasPose()) { spdlog::error("Image {} has no pose.", point_in_image2.image_id); exit(1); }

      constraint_type = TryTriangulate(config, image1, image2, camera1, camera2, point_on_image1, point_in_image2, kdtree, point_cloud, iter, point3D,
                                       lidar_point, lidar_normal);
      if (constraint_type != xcolor::TrackConstraintType::kVisualAndLidar) {
        continue;
      }

#pragma omp critical
      {
        if (++count % 40000 == 0) {
          spdlog::info("Triangulate point {}", count);
        }

        optimized_image_ids.insert(point_on_image1.image_id);
        optimized_image_ids.insert(point_in_image2.image_id);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image1.point_pixel, image1, camera1, loss_function_image, residual_block_ids);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_in_image2.point_pixel, image2, camera2, loss_function_image, residual_block_ids);
        xcolor::AddLidarFactorToProblem(problem, point3D, lidar_point, lidar_normal, loss_function_lidar, lidar_residual_block_ids);
      }
    }
    spdlog::info("{} images are used.", optimized_image_ids.size());
    spdlog::info("{} images in total.", images.size());
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    PrintMatchTrackStatistics(match_tracks);
    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_twoview[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_twoview[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, output_path + "/triangulated" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout       = true;
    options.num_threads                        = config.ba_optimization_num_threads;
    options.max_num_iterations                 = 20;
    options.linear_solver_type                 = ceres::SPARSE_SCHUR;
    options.preconditioner_type                = ceres::CLUSTER_TRIDIAGONAL;
    options.dense_linear_algebra_library_type  = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    ceres::Solve(options, &problem, &summary);
    spdlog::info("{}", summary.FullReport());

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_twoview[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_twoview[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, output_path + "/triangulated" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(output_path + "/image-poses.txt", output_path + "/images.bin", optimized_image_ids, images, pose_priors);
    SaveCameraParams(output_path + "/camera-params.txt", output_path + "/cameras.bin", cameras);
  }
}

void RunMultipleViewBA(const xcolor::SfmConfig &config, const std::string &output_path, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                       const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud,
                       const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                       std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                       std::vector<xcolor::MatchTrack> &match_tracks) {
  for (int iter = 0; iter < config.outer_opt_num_multi_view; ++iter) {
    ceres::Problem problem;
    auto loss_function_image = new ceres::CauchyLoss(config.reproject_cauchy_loss_scale);
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    std::vector<ceres::ResidualBlockId> lidar_residual_block_ids;

    std::unordered_set<colmap::image_t> optimized_image_ids;

    std::vector<colmap::Point3D> points3D;

    for (int i = 0; i < match_tracks.size(); ++i) {
      auto &match_pair = match_tracks[i];
      for (int j = 0; j < match_pair.point2D_on_imageN.size(); ++j) {
        auto &point_on_image = *match_pair.point2D_on_imageN[j];
        auto &image          = images[point_on_image.image_id];
        image.ResetPoint3DForPoint2D(point_on_image.point2D_idx);
      }
    }

    xcolor::Histogram average_depths;
    int count = 0;
#pragma omp parallel for
    for (int i = 0; i < match_tracks.size(); ++i) {
      auto &match_pair = match_tracks[i];
      auto &point3D    = match_tracks[i].point3D.point3D;

      int k = 8;
      std::vector<float> k_distances;
      std::vector<int> k_indices;
      kdtree.nearestKSearch(pcl::PointXYZINormal{(float)point3D.x(), (float)point3D.y(), (float)point3D.z()}, k, k_indices, k_distances);
      if (k_indices.size() != k) {
        spdlog::error("CHECK_EQ failed: k_indices.size() == k");
        exit(1);
      }

      Eigen::Vector3d surfel_center;
      Eigen::Vector3d surfel_normal;
      double surfel_std;
      ComputeSurfel(point_cloud, k_indices, surfel_center, surfel_normal, surfel_std);

#pragma omp critical
      {
        if (++count % 10000 == 0) {
          spdlog::info("Triangulate point {}", count);
        }

        std::vector<int> valid_indices;
        for (int j = 0; j < match_pair.point2D_on_imageN.size(); ++j) {
          auto &point_on_image = *match_pair.point2D_on_imageN[j];
          auto &image          = images[point_on_image.image_id];
          auto &camera         = cameras[point_on_image.camera_id];
          if (!image.HasPose()) { spdlog::error("Image {} has no pose.", point_on_image.image_id); exit(1); }
          // if the point is not visible in the image, ignore it
          if (xcolor::ComputePixelError(point3D, point_on_image.point_pixel, image, camera).norm() >
              config.reproject_error_outlier_thresholds_multiview[iter]) {
            continue;
          }
          valid_indices.push_back(j);
        }

        if (valid_indices.size() < config.min_track_len) {
          match_tracks[i].constraint_type = xcolor::TrackConstraintType::kUnconstrained;
          // DLOG(INFO) << "Match " << i << " has only " << valid_indices.size() << " valid points, skip it.";
        } else {
          double average_depth = 0;
          colmap::Point3D np;
          np.xyz = point3D;
          np.color.setConstant(0);
          np.color[2] = 255;
          np.error    = 0;
          for (int j = 0; j < valid_indices.size(); ++j) {
            auto &point_on_image = *match_pair.point2D_on_imageN[valid_indices[j]];
            auto &image          = images[point_on_image.image_id];
            auto &camera         = cameras[point_on_image.camera_id];
            optimized_image_ids.insert(point_on_image.image_id);
            xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image.point_pixel, image, camera, loss_function_image, residual_block_ids);

            average_depth += (colmap::Inverse(image.CamFromWorld()).translation - match_pair.point3D.point3D).norm() / valid_indices.size();

            np.track.AddElement(colmap::TrackElement(point_on_image.image_id, point_on_image.point2D_idx));

            image.SetPoint3DForPoint2D(point_on_image.point2D_idx, points3D.size());
          }

          average_depths.Add(average_depth);

          points3D.push_back(np);

          auto lidar_error = xcolor::ComputeLidarError(point3D, surfel_center, surfel_normal);
          if (std::abs(lidar_error) < config.lidar_error_outlier_thresholds_multiview[iter]) {
            double weight_by_depth;
            if (config.enable_weight_by_depth) {
              double alpha    = -log(0.2) / (40 * 40);
              weight_by_depth = exp(-alpha * average_depth * average_depth);
            } else {
              weight_by_depth = 1.0;
            }

            double weight            = 1 / sqrt(pow(0.05 / 6, 2) + surfel_std * surfel_std);
            auto loss_function_lidar = new ceres::ScaledLoss(new ceres::CauchyLoss(config.lidar_cauchy_weight),
                                                             weight_by_depth * weight * config.lidar_weight_scale, ceres::TAKE_OWNERSHIP);
            xcolor::AddLidarFactorToProblem(problem, point3D, surfel_center, surfel_normal, loss_function_lidar, lidar_residual_block_ids);

            match_tracks[i].constraint_type = xcolor::TrackConstraintType::kVisualAndLidar;
          } else {
            match_tracks[i].constraint_type = xcolor::TrackConstraintType::kVisualOnly;
          }
        }
      }
    }
    spdlog::info("{} images are used.", optimized_image_ids.size());
    spdlog::info("{} images in total.", images.size());
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    PrintMatchTrackStatistics(match_tracks);
    spdlog::info("Average depth: {}", average_depths.ToString(20));
    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, output_path + "/triangulated-multi" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout       = true;
    options.num_threads                        = config.ba_optimization_num_threads;
    options.max_num_iterations                 = 20;
    options.linear_solver_type                 = ceres::SPARSE_SCHUR;
    options.preconditioner_type                = ceres::CLUSTER_TRIDIAGONAL;
    options.dense_linear_algebra_library_type  = ceres::CUDA;
    options.sparse_linear_algebra_library_type = ceres::CUDA_SPARSE;
    ceres::Solve(options, &problem, &summary);
    spdlog::info("{}", summary.FullReport());

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_multiview[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, output_path + "/triangulated-multi" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(output_path + "/image-poses.txt", output_path + "/images.bin", optimized_image_ids, images, pose_priors);
    SaveCameraParams(output_path + "/camera-params.txt", output_path + "/cameras.bin", cameras);

    xcolor::WritePoints3DBinary(output_path + "/points3D.bin", points3D);
  }
}

}  // namespace xcolor
