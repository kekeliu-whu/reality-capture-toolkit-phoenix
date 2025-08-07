
#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/geometry/triangulation.h>
#include <colmap/scene/database.h>
#include <colmap/scene/database_cache.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/scene/reconstruction_io.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <rapidjson/document.h>
#include <rapidjson/filereadstream.h>
#include <rapidjson/rapidjson.h>
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>

#include "common/histogram.h"
#include "core/xsfm_lib.h"
#include "io/read_write.h"
#include "io/xml_io.h"

DEFINE_string(point_cloud_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/colorized.las_normals.pcd", "Point cloud filename");
DEFINE_string(point_cloud_offset_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/colorized.las_offset.csv", "Point cloud offset filename");
DEFINE_string(database_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/xsfm.db", "Database filename");
DEFINE_string(initial_pose_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/images/ImgPose.txt", "Initial pose filename");
DEFINE_string(calibration_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/calibration.yaml", "");
DEFINE_string(images_path, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/images", "");

DEFINE_string(output_path, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm", "Output path");

void SaveTriangulatedPoints(const std::vector<xcolor::MatchTrack> &match_tracks, const std::string &filename) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud_out;
  for (int i = 0; i < match_tracks.size(); ++i) {
    if (match_tracks[i].point3D.valid) {
      auto point3D = match_tracks[i].point3D.point3D;
      pcl::PointXYZINormal np;
      np.x = point3D.x();
      np.y = point3D.y();
      np.z = point3D.z();
      point_cloud_out.push_back(np);
    }
  }
  DLOG(INFO) << "savePCDFileBinary: " << pcl::io::savePCDFileBinary(filename, point_cloud_out);
}

void SaveImagePoses(const std::string &filename, const std::unordered_set<colmap::image_t> &optimized_image_ids,
                    const std::unordered_map<colmap::image_t, colmap::Image> &images,
                    const std::unordered_map<colmap::camera_t, colmap::Rigid3d> &pose_priors) {
  DLOG(INFO) << "Saving image poses into " << FLAGS_output_path + "/images.bin";
  std::vector<colmap::Image> images_out;
  for (auto &image_id : optimized_image_ids) {
    auto &image          = images.at(image_id);
    auto &pose_prior_w2c = pose_priors.at(image.ImageId());
    auto pose_c2w        = colmap::Inverse(image.CamFromWorld());
    double distance      = (pose_prior_w2c * pose_c2w).translation.norm();
    if (distance > 0.35) {
      continue;
    }
    images_out.push_back(images.at(image_id));
  }
  LOG_IF(WARNING, optimized_image_ids.size() != images_out.size()) << images_out.size() << " / " << optimized_image_ids.size() << " poses is valid";
  xcolor::WriteImagesBinary(FLAGS_output_path + "/images.bin", images_out);

  DLOG(INFO) << "Saving image poses into " << filename;
  std::ofstream infile(filename);
  infile << "camera_id image_name x y z rw rx ry rz" << std::endl;
  for (auto &image_id : optimized_image_ids) {
    auto &image          = images.at(image_id);
    auto &pose_prior_w2c = pose_priors.at(image.ImageId());
    auto pose_c2w        = colmap::Inverse(image.CamFromWorld());
    double distance      = (pose_prior_w2c * pose_c2w).translation.norm();
    if (distance > 0.35) {
      continue;
    }
    infile << std::fixed << std::setprecision(6) << image.CameraId() << " " << image.Name() << " " << pose_c2w.translation.x() << " "
           << pose_c2w.translation.y() << " " << pose_c2w.translation.z() << " " << pose_c2w.rotation.w() << " " << pose_c2w.rotation.x() << " "
           << pose_c2w.rotation.y() << " " << pose_c2w.rotation.z() << std::endl;
  }
}

void SaveCameraParams(const std::string &filename, const std::unordered_map<colmap::camera_t, colmap::Camera> &cameras) {
  DLOG(INFO) << "Saving camera params into " << FLAGS_output_path + "/images.bin";
  std::vector<colmap::Camera> cameras_out;
  for (auto &[image_id, _] : cameras) {
    cameras_out.push_back(cameras.at(image_id));
  }
  xcolor::WriteCamerasBinary(FLAGS_output_path + "/cameras.bin", cameras_out);

  DLOG(INFO) << "Saving camera params into " << filename;
  std::ofstream infile(filename);
  infile << "camera_id model_id fx fy cx cy params..." << std::endl;
  for (auto &e : cameras) {
    infile << std::fixed << std::setprecision(6) << e.first << " " << (int)e.second.model_id << " " << e.second.params[0] << " " << e.second.params[0]
           << " " << e.second.params[1] << " " << e.second.params[2] << " " << e.second.params[3] << " " << e.second.params[4] << " "
           << e.second.params[5] << " " << e.second.params[6] << std::endl;
  }
}

bool TryTriangulate(const xcolor::SfmConfig &config, const colmap::Image &image1, const colmap::Image &image2, const colmap::Camera &camera1,
                    const colmap::Camera &camera2, const xcolor::Point2DInfo &point_on_image1, const xcolor::Point2DInfo &point_in_image2,
                    const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree, const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, int iter,
                    Eigen::Vector3d &point3D, Eigen::Vector3d &lidar_point, Eigen::Vector3d &lidar_normal) {
  auto cam_from_world1 = image1.CamFromWorld().ToMatrix();
  auto cam_from_world2 = image2.CamFromWorld().ToMatrix();

  // if colmap::TriangulatePoint failed, ignore the match
  bool ok = colmap::TriangulatePoint(cam_from_world1, cam_from_world2, camera1.CamFromImg(point_on_image1.point_pixel),
                                     camera2.CamFromImg(point_in_image2.point_pixel), &point3D);
  if (!ok) {
    return false;
  }

  Eigen::Vector3d cam1_center = -(cam_from_world1.block<3, 3>(0, 0).transpose() * cam_from_world1.block<3, 1>(0, 3));
  Eigen::Vector3d cam2_center = -(cam_from_world2.block<3, 3>(0, 0).transpose() * cam_from_world2.block<3, 1>(0, 3));

  // if the viewing angle between the two perspectives is too small, the match will be ignored
  double angle = colmap::CalculateTriangulationAngle(cam1_center, cam2_center, point3D);
  if (std::abs(angle) < config.min_tri_angle * M_PI / 180) {
    return false;
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
    return false;
  }

  // if reprojection error is too large, the match will be ignored
  auto projection_error1 = xcolor::ComputePixelError(point3D, point_on_image1.point_pixel, image1, camera1);
  auto projection_error2 = xcolor::ComputePixelError(point3D, point_in_image2.point_pixel, image2, camera2);
  if (projection_error1.norm() > config.reproject_error_outlier_thresholds_twoview[iter] ||
      projection_error2.norm() > config.reproject_error_outlier_thresholds_twoview[iter]) {
    return false;
  }

  return true;
}

void ComputeSurfel(const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, const std::vector<int> &k_indices, Eigen::Vector3d &surfel_center,
                   Eigen::Vector3d &surfel_normal, double &surfel_std) {
  int k = k_indices.size();

  Eigen::Vector3d center = Eigen::Vector3d::Zero();
  for (int i = 0; i < k; i++) {
    center += point_cloud[k_indices[i]].getVector3fMap().cast<double>();
  }
  center /= k;

  Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
  for (int i = 0; i < k; i++) {
    Eigen::Vector3d diff = point_cloud[k_indices[i]].getVector3fMap().cast<double>() - center;
    covariance += diff * diff.transpose();
  }
  covariance /= k;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
  Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();
  Eigen::Vector3d normal       = eigenvectors.col(0);

  surfel_center = center;
  surfel_normal = normal;
  surfel_std    = std::sqrt(std::max(0.0, eigensolver.eigenvalues()[0]));  // standard deviation along the normal direction
}

void RunMultipleViewBA(const xcolor::SfmConfig &config, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
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
      CHECK_EQ(k_indices.size(), k);

      Eigen::Vector3d surfel_center;
      Eigen::Vector3d surfel_normal;
      double surfel_std;
      ComputeSurfel(point_cloud, k_indices, surfel_center, surfel_normal, surfel_std);

#pragma omp critical
      {
        if (++count % 10000 == 0) {
          DLOG(INFO) << "Triangulate point " << count;
        }

        std::vector<int> valid_indices;
        for (int j = 0; j < match_pair.point2D_on_imageN.size(); ++j) {
          auto &point_on_image = *match_pair.point2D_on_imageN[j];
          auto &image          = images[point_on_image.image_id];
          auto &camera         = cameras[point_on_image.camera_id];
          CHECK(image.HasPose()) << "Image " << point_on_image.image_id << " has no pose.";
          // if the point is not visible in the image, ignore it
          if (xcolor::ComputePixelError(point3D, point_on_image.point_pixel, image, camera).norm() >
              config.reproject_error_outlier_thresholds_multiview[iter]) {
            continue;
          }
          valid_indices.push_back(j);
        }

        if (valid_indices.size() < config.min_track_len) {
          match_tracks[i].point3D.valid = 0;
          DLOG(INFO) << "Match " << i << " has only " << valid_indices.size() << " valid points, skip it.";
        } else {
          match_tracks[i].point3D.valid = 1;

          double average_depth = 0;
          for (int j = 0; j < valid_indices.size(); ++j) {
            auto &point_on_image = *match_pair.point2D_on_imageN[valid_indices[j]];
            auto &image          = images[point_on_image.image_id];
            auto &camera         = cameras[point_on_image.camera_id];
            optimized_image_ids.insert(point_on_image.image_id);
            xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image.point_pixel, image, camera, loss_function_image, residual_block_ids);

            average_depth += (colmap::Inverse(image.CamFromWorld()).translation - match_pair.point3D.point3D).norm() / valid_indices.size();
          }

          average_depths.Add(average_depth);

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
          }
        }
      }
    }
    DLOG(INFO) << optimized_image_ids.size() << " images are used.";
    DLOG(INFO) << images.size() << " images in total.";
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    int valid_count = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &a) { return sum + a.point3D.valid; });
    DLOG(INFO) << valid_count << "/" << match_tracks.size() << " (" << valid_count * 100.0 / match_tracks.size() << "%) matches are valid.";

    //DLOG(INFO) << "Average depth: " << average_depths.ToString(20);
    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated-multi" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads                  = config.ba_optimization_num_threads;
    options.max_num_iterations           = 20;
    ceres::Solve(options, &problem, &summary);
    DLOG(INFO) << summary.FullReport();

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_multiview[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_multiview[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated-multi" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(FLAGS_output_path + "/image-poses.txt", optimized_image_ids, images, pose_priors);
    SaveCameraParams(FLAGS_output_path + "/camera-params.txt", cameras);
  }
}

void RunTwoViewBA(const xcolor::SfmConfig &config, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
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
      Eigen::Vector3d lidar_point;
      Eigen::Vector3d lidar_normal;

      CHECK(image1.HasPose()) << "Image " << point_on_image1.image_id << " has no pose.";
      CHECK(image2.HasPose()) << "Image " << point_in_image2.image_id << " has no pose.";

      bool valid = TryTriangulate(config, image1, image2, camera1, camera2, point_on_image1, point_in_image2, kdtree, point_cloud, iter, point3D,
                                  lidar_point, lidar_normal);
      if (!valid) {
        continue;
      }

      match_tracks[i].point3D.valid = 1;

#pragma omp critical
      {
        if (++count % 40000 == 0) {
          DLOG(INFO) << "Triangulate point " << count;
        }

        optimized_image_ids.insert(point_on_image1.image_id);
        optimized_image_ids.insert(point_in_image2.image_id);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image1.point_pixel, image1, camera1, loss_function_image, residual_block_ids);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_in_image2.point_pixel, image2, camera2, loss_function_image, residual_block_ids);
        xcolor::AddLidarFactorToProblem(problem, point3D, lidar_point, lidar_normal, loss_function_lidar, lidar_residual_block_ids);
      }
    }
    DLOG(INFO) << optimized_image_ids.size() << " images are used.";
    DLOG(INFO) << images.size() << " images in total.";
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    int valid_count = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &a) { return sum + a.point3D.valid; });
    DLOG(INFO) << valid_count << "/" << match_tracks.size() << " (" << valid_count * 100.0 / match_tracks.size() << "%) matches are valid.";

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_twoview[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_twoview[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads                  = config.ba_optimization_num_threads;
    options.max_num_iterations           = 20;
    ceres::Solve(options, &problem, &summary);
    DLOG(INFO) << summary.FullReport();

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds_twoview[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds_twoview[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(FLAGS_output_path + "/image-poses.txt", optimized_image_ids, images, pose_priors);
    SaveCameraParams(FLAGS_output_path + "/camera-params.txt", cameras);
  }
}

void RunSFM(const xcolor::SfmConfig &config, const std::string &point_cloud_filename, std::unordered_map<colmap::image_t, colmap::Image> &images,
            std::unordered_map<colmap::camera_t, colmap::Camera> &cameras, std::vector<xcolor::MatchTrack> &match_tracks_coarse,
            std::vector<xcolor::MatchTrack> &match_tracks_fine) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
  int load_ply_status = pcl::io::loadPCDFile(point_cloud_filename, point_cloud);
  CHECK_NE(load_ply_status, -1);
  DLOG(INFO) << "Load " << point_cloud.size() << " lidar points.";

  // build kdtree
  pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
  kdtree.setInputCloud(point_cloud.makeShared());

  std::unordered_map<colmap::image_t, colmap::Rigid3d> pose_priors;
  for (auto &[image_id, image] : images) {
    pose_priors[image.ImageId()] = image.CamFromWorld();
  }

  RunTwoViewBA(config, kdtree, point_cloud, pose_priors, images, cameras, match_tracks_coarse);

  MergeTrack(match_tracks_coarse, match_tracks_fine, config.use_all_track, config.min_track_len);

  RunMultipleViewBA(config, kdtree, point_cloud, pose_priors, images, cameras, match_tracks_fine);

  {
    std::vector<colmap::Point3D> points3D;
    points3D.reserve(point_cloud.size());
    for (auto &p : point_cloud) {
      colmap::Point3D np;
      np.xyz = p.getVector3fMap().cast<double>();
      np.color.setConstant(255);
      np.error = 0;
      points3D.push_back(np);
    }
    xcolor::WritePoints3DBinary(FLAGS_output_path + "/points3D.bin", points3D);
  }
}

std::unordered_map<std::string, colmap::Rigid3d> ReadImagePoses(const std::string &filename, const Eigen::Vector2d &offset) {
  std::unordered_map<std::string, colmap::Rigid3d> image_to_pose;

  std::ifstream file(filename);
  CHECK(file) << filename;

  std::string line;
  std::getline(file, line);

  std::string file_path;
  double tx, ty, tz, roll, pitch, yaw, qx, qy, qz, qw, timestamp;
  std::map<std::string, std::optional<colmap::Rigid3d>> image_to_last_pose_map;
  while (file >> file_path >> tx >> ty >> tz >> roll >> pitch >> yaw >> qx >> qy >> qz >> qw >> timestamp) {
    colmap::Rigid3d pose_world_from_cv;
    pose_world_from_cv.translation     = Eigen::Vector3d(tx - offset.x(), ty - offset.y(), tz);
    pose_world_from_cv.rotation        = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    colmap::Rigid3d pose_cv_from_world = colmap::Inverse(pose_world_from_cv);

    std::string camera_dirname = boost::filesystem::path(file_path).parent_path().string();

    auto &camera_to_last_pose = image_to_last_pose_map[camera_dirname];
    if (camera_to_last_pose.has_value()) {
      if (pose_cv_from_world.rotation.angularDistance(camera_to_last_pose->rotation) * 180 / M_PI > 1.0 ||
          (pose_cv_from_world.translation - camera_to_last_pose->translation).norm() > 0.3) {
        camera_to_last_pose      = pose_cv_from_world;
        image_to_pose[file_path] = pose_cv_from_world;
      } else {
        DLOG(INFO) << "Ignore image " << file_path;
      }
    } else {
      camera_to_last_pose      = pose_cv_from_world;
      image_to_pose[file_path] = pose_cv_from_world;
    }
  }

  DLOG(INFO) << "Read " << image_to_pose.size() << " image poses from " << filename;
  return image_to_pose;
}

void ReadPointCloudOffset(const std::string &filename, Eigen::Vector2d &offset, double &lon, double &lat) {
  std::ifstream infile(filename);
  if (infile.is_open()) {
    std::string line;
    std::getline(infile, line);  // Skip the header line
    if (std::getline(infile, line)) {
      std::replace(line.begin(), line.end(), ',', ' ');
      std::istringstream iss(line);
      iss >> offset.x() >> offset.y() >> lon >> lat;
    }
    infile.close();
  } else {
    DLOG(ERROR) << "Failed to open file: " << filename;
  }
}

void SaveXml(const std::string &filename, const std::unordered_map<colmap::image_t, colmap::Image> &images,
             const std::unordered_map<colmap::camera_t, colmap::Camera> &cameras, std::vector<xcolor::MatchTrack> &match_tracks, double longitude,
             double latitude) {
  BlocksExchange be;

  SpatialReferenceSystem srs;
  srs.id         = 0;
  srs.name       = (boost::format("Local East-North-Up(ENU); origin: %.9fN %.9fE") % latitude % longitude).str();
  srs.definition = (boost::format("ENU:%.9f,%.9f") % latitude % longitude).str();
  be.addSpatialReferenceSystem(srs);

  be.block.srs_id = srs.id;

  for (auto &[camera_id, camera] : cameras) {
    std::unordered_map<colmap::image_t, colmap::Image> select_images;
    for (auto &[image_id, image] : images) {
      if (image.CameraId() == camera_id) {
        select_images[image_id] = image;
      }
    }
    if (select_images.empty()) {
      continue;
    }

    Photogroup pg;
    pg.aspect_ratio            = 1.0;
    pg.camera_model_type       = "Perspective";
    pg.camera_orientation      = "XRightYDown";
    pg.distortion.k1           = camera.params[3];
    pg.distortion.k2           = camera.params[4];
    pg.distortion.p1           = camera.params[5];
    pg.distortion.p2           = camera.params[6];
    pg.distortion.k3           = 0;
    pg.principal_point.x       = camera.PrincipalPointX();
    pg.principal_point.y       = camera.PrincipalPointY();
    pg.sensor_size             = 24;
    pg.focal_length_pixels     = camera.FocalLength();
    pg.focal_length            = camera.FocalLength() / std::max(camera.width, camera.height) * pg.sensor_size;
    pg.image_dimensions.width  = camera.width;
    pg.image_dimensions.height = camera.height;
    for (auto &[image_id, image] : select_images) {
      Photo photo;
      photo.id                 = image_id;
      photo.far_depth          = 200;
      photo.near_depth         = 2;
      photo.median_depth       = 40;
      photo.component          = camera_id;
      photo.image_path         = FLAGS_images_path + "/" + image.Name();
      auto pos                 = colmap::Inverse(image.CamFromWorld()).translation;
      auto rot                 = image.CamFromWorld().rotation.matrix();
      photo.pose.center.x      = pos.x();
      photo.pose.center.y      = pos.y();
      photo.pose.center.z      = pos.z();
      photo.pose.rotation.m_00 = rot(0, 0);
      photo.pose.rotation.m_01 = rot(0, 1);
      photo.pose.rotation.m_02 = rot(0, 2);
      photo.pose.rotation.m_10 = rot(1, 0);
      photo.pose.rotation.m_11 = rot(1, 1);
      photo.pose.rotation.m_12 = rot(1, 2);
      photo.pose.rotation.m_20 = rot(2, 0);
      photo.pose.rotation.m_21 = rot(2, 1);
      photo.pose.rotation.m_22 = rot(2, 2);
      pg.photos.push_back(photo);
    }
    be.block.photogroups.push_back(pg);
  }

  for (auto &track : match_tracks) {
    TiePoint tp;
    tp.color.blue  = 1;
    tp.color.green = 1;
    tp.color.red   = 1;
    tp.position.x  = track.point3D.point3D.x();
    tp.position.y  = track.point3D.point3D.y();
    tp.position.z  = track.point3D.point3D.z();
    std::set<colmap::image_t> image_ids;
    for (auto &point2D_on_image : track.point2D_on_imageN) {
      if (image_ids.find(point2D_on_image->image_id) != image_ids.end()) {
        DLOG(WARNING) << "dup point2D";
        continue;
      }
      image_ids.insert(point2D_on_image->image_id);

      Measurement mea;
      mea.photo_id = point2D_on_image->image_id;
      mea.x        = point2D_on_image->point_pixel.x();
      mea.y        = point2D_on_image->point_pixel.y();
      tp.measurements.push_back(mea);
    }
    be.addTiePoint(tp);
  }

  be.saveToXML(filename);
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);
  // EncryptedLogSink *sink = new EncryptedLogSink();
  // google::AddLogSink(sink);

  FLAGS_logtostderr = 1;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  xcolor::SfmConfig config{cores_used};

  std::vector<xcolor::MatchTrack> match_tracks_coarse;
  std::unordered_map<colmap::image_t, colmap::Image> images;
  std::unordered_map<colmap::camera_t, colmap::Camera> cameras;

  // reload images
  double longitude, latitude;
  {
    Eigen::Vector2d offset;
    ReadPointCloudOffset(FLAGS_point_cloud_offset_filename, offset, longitude, latitude);
    std::unordered_map<std::string, colmap::Rigid3d> image_to_pose = ReadImagePoses(FLAGS_initial_pose_filename, offset);

    std::unordered_set<std::string> image_names;
    for (const auto &pair : image_to_pose) {
      image_names.insert(pair.first);
    }

    colmap::Database database(FLAGS_database_filename);
    auto database_cache    = colmap::DatabaseCache::Create(database, config.min_num_matches, config.ignore_watermarks, image_names);
    cameras                = database_cache->Cameras();
    images                 = database_cache->Images();
    const auto &corr_graph = *database_cache->CorrespondenceGraph();

    for (auto &image : images) {
      image.second.SetCamFromWorld(image_to_pose[image.second.Name()]);
    }

    // get image count with pose
    int image_count = 0;
    for (auto &image : images) {
      if (image.second.HasPose()) {
        image_count++;
      }
    }
    DLOG(INFO) << "Image count with pose: " << image_count << "/" << images.size() << " (" << image_count * 100.0 / images.size() << "%)";

    match_tracks_coarse = GenerateMatchPairs(corr_graph, images, config);
    DLOG(INFO) << "Loading " << match_tracks_coarse.size() << " image pairs from " << FLAGS_database_filename;
  }

  std::vector<xcolor::MatchTrack> match_tracks_fine;
  RunSFM(config, FLAGS_point_cloud_filename, images, cameras, match_tracks_coarse, match_tracks_fine);
  SaveXml(FLAGS_output_path + "/mvs.xml", images, cameras, match_tracks_fine, longitude, latitude);

  DLOG(INFO) << "done.";
  std::cout << "done." << std::endl;

  google::ShutdownGoogleLogging();
  return 0;
}