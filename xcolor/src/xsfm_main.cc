
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

#include "common/histogram.h"
#include "core/xsfm_lib.h"
#include "io/read_write.h"

DEFINE_string(point_cloud_filename, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/colorized.las_normals.pcd",
              "Point cloud filename");
DEFINE_string(output_path, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f", "Output path");
DEFINE_string(database_filename, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/xsfm.db", "Database filename");
DEFINE_string(initial_pose_dirname, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/", "Initial pose filename");
DEFINE_int32(pose_type, 1, "Pose type, =0 for s10, =1 for s20, =2 for export s20 poses");
//////////////////////// only used when pose_type = 2 begin ////////////////////////
DEFINE_string(calibration_filename, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/calibration.yaml", "");
DEFINE_string(images_path, "D:/project_3d/data/sfm-share/output_dir_s20_first-oldest-outdoor-shareuav1f/undistort", "");
//////////////////////// only used when pose_type = 2 end   ////////////////////////

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
  DLOG(INFO) << pcl::io::savePCDFileBinary(filename, point_cloud_out);
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
    infile << std::fixed << std::setprecision(6) << e.first << " " << (int)e.second.model_id << " " << e.second.params[0] << " " << e.second.params[1]
           << " " << e.second.params[2] << " " << e.second.params[3] << " " << e.second.params[4] << " " << e.second.params[5] << " "
           << e.second.params[6] << " " << e.second.params[7] << std::endl;
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
  if (std::abs(lidar_error) > config.lidar_error_outlier_thresholds[iter]) {
    return false;
  }

  // if reprojection error is too large, the match will be ignored
  auto projection_error1 = xcolor::ComputePixelError(point3D, point_on_image1.point_pixel, image1, camera1);
  auto projection_error2 = xcolor::ComputePixelError(point3D, point_in_image2.point_pixel, image2, camera2);
  if (projection_error1.norm() > config.reproject_error_outlier_thresholds[iter] ||
      projection_error2.norm() > config.reproject_error_outlier_thresholds[iter]) {
    return false;
  }

  return true;
}

void RunMultipleViewBA(const xcolor::SfmConfig &config, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                       const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud,
                       const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                       std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                       std::vector<xcolor::MatchTrack> &match_tracks) {
  for (int iter = 0; iter < config.outer_opt_num; ++iter) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem{problem_options};
    auto loss_function_image = std::make_shared<ceres::CauchyLoss>(config.reproject_cauchy_loss_scale);
    auto loss_function_lidar = std::make_shared<ceres::ScaledLoss>(new ceres::CauchyLoss(config.lidar_cauchy_loss_scale), config.lidar_loss_scale,
                                                                   ceres::DO_NOT_TAKE_OWNERSHIP);
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    std::vector<ceres::ResidualBlockId> lidar_residual_block_ids;

    std::unordered_set<colmap::image_t> optimized_image_ids;

    int count = 0;
#pragma omp parallel for
    for (int i = 0; i < match_tracks.size(); ++i) {
      auto &match_pair = match_tracks[i];
      auto &point3D    = match_tracks[i].point3D.point3D;

      Eigen::Vector3d lidar_point;
      Eigen::Vector3d lidar_normal;
      std::vector<float> distances;
      std::vector<int> indices;
      kdtree.nearestKSearch(pcl::PointXYZINormal{(float)point3D.x(), (float)point3D.y(), (float)point3D.z()}, 1, indices, distances);

      auto nearest_point = point_cloud[indices[0]];
      lidar_point        = nearest_point.getVector3fMap().cast<double>();
      lidar_normal       = nearest_point.getNormalVector3fMap().cast<double>();

      match_tracks[i].point3D.valid = 1;

#pragma omp critical
      {
        if (++count % 10000 == 0) {
          DLOG(INFO) << "Triangulate point " << count;
        }

        for (int k = 0; k < match_pair.point2D_on_imageN.size(); ++k) {
          auto &point_on_image = *match_pair.point2D_on_imageN[k];
          auto &image          = images[point_on_image.image_id];
          auto &camera         = cameras[point_on_image.camera_id];
          CHECK(image.HasPose()) << "Image " << point_on_image.image_id << " has no pose.";
          optimized_image_ids.insert(point_on_image.image_id);
          xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image.point_pixel, image, camera, loss_function_image.get(),
                                              residual_block_ids);
        }
        xcolor::AddLidarFactorToProblem(problem, point3D, lidar_point, lidar_normal, loss_function_lidar.get(), lidar_residual_block_ids);
      }
    }
    DLOG(INFO) << optimized_image_ids.size() << " images are used.";
    DLOG(INFO) << images.size() << " images in total.";
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    int valid_count = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &a) { return sum + a.point3D.valid; });
    DLOG(INFO) << valid_count << "/" << match_tracks.size() << " (" << valid_count * 100.0 / match_tracks.size() << "%) matches are valid.";

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated-multi" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads                  = config.ba_optimization_num_threads;
    options.max_num_iterations           = 20;
    ceres::Solve(options, &problem, &summary);
    DLOG(INFO) << summary.FullReport();

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated-multi" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(FLAGS_output_path + "/image-poses.txt", optimized_image_ids, images, pose_priors);
    SaveCameraParams(FLAGS_output_path + "/camera-params.txt", cameras);
  }
}

void RunTwoViewBA(const xcolor::SfmConfig &config, const pcl::KdTreeFLANN<pcl::PointXYZINormal> &kdtree,
                  const pcl::PointCloud<pcl::PointXYZINormal> &point_cloud, const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                  std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras,
                  std::vector<xcolor::MatchTrack> &match_tracks) {
  for (int iter = 0; iter < config.outer_opt_num; ++iter) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem{problem_options};
    auto loss_function_image = std::make_shared<ceres::CauchyLoss>(config.reproject_cauchy_loss_scale);
    auto loss_function_lidar = std::make_shared<ceres::ScaledLoss>(new ceres::CauchyLoss(config.lidar_cauchy_loss_scale), config.lidar_loss_scale,
                                                                   ceres::DO_NOT_TAKE_OWNERSHIP);
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
        if (++count % 10000 == 0) {
          DLOG(INFO) << "Triangulate point " << count;
        }

        optimized_image_ids.insert(point_on_image1.image_id);
        optimized_image_ids.insert(point_in_image2.image_id);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_on_image1.point_pixel, image1, camera1, loss_function_image.get(),
                                            residual_block_ids);
        xcolor::AddReprojectFactorToProblem(problem, point3D, point_in_image2.point_pixel, image2, camera2, loss_function_image.get(),
                                            residual_block_ids);
        xcolor::AddLidarFactorToProblem(problem, point3D, lidar_point, lidar_normal, loss_function_lidar.get(), lidar_residual_block_ids);
      }
    }
    DLOG(INFO) << optimized_image_ids.size() << " images are used.";
    DLOG(INFO) << images.size() << " images in total.";
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    int valid_count = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &a) { return sum + a.point3D.valid; });
    DLOG(INFO) << valid_count << "/" << match_tracks.size() << " (" << valid_count * 100.0 / match_tracks.size() << "%) matches are valid.";

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds[iter], problem, residual_block_ids, "image");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds[iter], problem, lidar_residual_block_ids, "lidar");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads                  = config.ba_optimization_num_threads;
    options.max_num_iterations           = 20;
    ceres::Solve(options, &problem, &summary);
    DLOG(INFO) << summary.FullReport();

    xcolor::PrintResidualHistogram(config.reproject_error_outlier_thresholds[iter], problem, residual_block_ids, "image_refined");
    xcolor::PrintResidualHistogram(config.lidar_error_outlier_thresholds[iter], problem, lidar_residual_block_ids, "lidar_refined");
    SaveTriangulatedPoints(match_tracks, FLAGS_output_path + "/triangulated" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(FLAGS_output_path + "/image-poses.txt", optimized_image_ids, images, pose_priors);
    SaveCameraParams(FLAGS_output_path + "/camera-params.txt", cameras);
  }
}

void RunSFM(const xcolor::SfmConfig &config, std::vector<xcolor::MatchTrack> &match_tracks, const std::string &point_cloud_filename,
            std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
  int load_ply_status = pcl::io::loadPCDFile(point_cloud_filename, point_cloud);
  CHECK_NE(load_ply_status, -1);
  DLOG(INFO) << "Load " << point_cloud.size() << " points.";

  // build kdtree
  pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
  kdtree.setInputCloud(point_cloud.makeShared());

  std::unordered_map<colmap::image_t, colmap::Rigid3d> pose_priors;
  for (auto &[image_id, image] : images) {
    pose_priors[image.ImageId()] = image.CamFromWorld();
  }

  RunTwoViewBA(config, kdtree, point_cloud, pose_priors, images, cameras, match_tracks);

  std::vector<xcolor::MatchTrack> match_tracks_merged;
  MergeTrack(match_tracks, match_tracks_merged);

  RunMultipleViewBA(config, kdtree, point_cloud, pose_priors, images, cameras, match_tracks_merged);

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

std::unordered_map<std::string, colmap::Rigid3d> ReadImagePosesType0(const std::string &filename) {
  std::unordered_map<std::string, colmap::Rigid3d> image_to_pose;

  std::ifstream file(filename);
  CHECK(file) << filename;

  std::string json_str{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

  rapidjson::Document doc;
  rapidjson::StringStream string_stream{json_str.c_str()};
  doc.ParseStream(string_stream);

  std::optional<colmap::Rigid3d> last_pose = std::nullopt;
  auto &frames                             = doc["frames"];
  for (int i = 0; i < frames.Size(); ++i) {
    auto &frame = frames[i];

    auto &mat = frame["transform_matrix"];
    Eigen::Matrix4d T;
    T << mat[0][0].GetDouble(), mat[0][1].GetDouble(), mat[0][2].GetDouble(), mat[0][3].GetDouble(), mat[1][0].GetDouble(), mat[1][1].GetDouble(),
        mat[1][2].GetDouble(), mat[1][3].GetDouble(), mat[2][0].GetDouble(), mat[2][1].GetDouble(), mat[2][2].GetDouble(), mat[2][3].GetDouble(), 0,
        0, 0, 1;

    colmap::Rigid3d pose_cg_to_world;
    pose_cg_to_world.rotation = T.block<3, 3>(0, 0);
    pose_cg_to_world.rotation.normalize();
    pose_cg_to_world.translation = T.block<3, 1>(0, 3);

    std::string file_path = frame["file_path"].GetString();
    auto file             = boost::filesystem::path(file_path);
    auto timestamp        = frame["timestamp"].GetUint64();

    auto real_filename = file.parent_path().string() + "/" + std::to_string(timestamp) + ".png";

    colmap::Rigid3d pose_cg_from_world = colmap::Inverse(pose_cg_to_world);

    colmap::Rigid3d pose_cv_from_world;

    Eigen::Matrix3d mat_cg_to_cv;
    mat_cg_to_cv << 1, 0, 0, 0, -1, 0, 0, 0, -1;
    pose_cv_from_world.translation = mat_cg_to_cv * pose_cg_from_world.translation;
    pose_cv_from_world.rotation    = Eigen::Quaterniond(mat_cg_to_cv) * pose_cg_from_world.rotation;
    if (last_pose.has_value()) {
      if (pose_cv_from_world.rotation.angularDistance(last_pose->rotation) * 180 / M_PI > 1.0 ||
          (pose_cv_from_world.translation - last_pose->translation).norm() > 0.3) {
        last_pose                    = pose_cv_from_world;
        image_to_pose[real_filename] = pose_cv_from_world;
      }
    } else {
      last_pose                    = pose_cv_from_world;
      image_to_pose[real_filename] = pose_cv_from_world;
    }
  }

  return image_to_pose;
}

std::unordered_map<std::string, colmap::Rigid3d> ReadImagePosesType1(const std::string &filename) {
  std::unordered_map<std::string, colmap::Rigid3d> image_to_pose;

  std::ifstream file(filename);
  CHECK(file) << filename;

  std::string line;
  std::getline(file, line);

  std::string index;
  double tx, ty, tz, qx, qy, qz, qw, timestamp;
  std::optional<colmap::Rigid3d> last_pose = std::nullopt;
  while (file >> index >> tx >> ty >> tz >> qx >> qy >> qz >> qw >> timestamp) {
    std::string file_path = (index.find("left") != std::string::npos ? "left/" : "right/") + index + ".jpg";
    colmap::Rigid3d pose_world_from_cv;
    pose_world_from_cv.translation     = Eigen::Vector3d(tx, ty, tz);
    pose_world_from_cv.rotation        = Eigen::Quaterniond(qw, qx, qy, qz).normalized();
    colmap::Rigid3d pose_cv_from_world = colmap::Inverse(pose_world_from_cv);
    if (last_pose.has_value()) {
      if (pose_cv_from_world.rotation.angularDistance(last_pose->rotation) * 180 / M_PI > 1.0 ||
          (pose_cv_from_world.translation - last_pose->translation).norm() > 0.3) {
        last_pose                = pose_cv_from_world;
        image_to_pose[file_path] = pose_cv_from_world;
      } else {
        DLOG(INFO) << "Ignore image " << file_path;
      }
    } else {
      last_pose                = pose_cv_from_world;
      image_to_pose[file_path] = pose_cv_from_world;
    }
  }

  return image_to_pose;
}

std::unordered_map<std::string, colmap::Rigid3d> ReadImagePoses(int type, const std::string &pose_dirname) {
  std::unordered_map<std::string, colmap::Rigid3d> ret;
  switch (type) {
    case 0: {
      ret = ReadImagePosesType0(pose_dirname + "/transforms.json");
    } break;
    case 1:
    case 2: {
      std::unordered_map<std::string, colmap::Rigid3d> ret_left  = ReadImagePosesType1(pose_dirname + "/leftImgPose.txt");
      std::unordered_map<std::string, colmap::Rigid3d> ret_right = ReadImagePosesType1(pose_dirname + "/rightImgPose.txt");
      ret.insert(ret_left.begin(), ret_left.end());
      ret.insert(ret_right.begin(), ret_right.end());
    } break;
    default: {
      DLOG(FATAL) << "Unknown pose type: " << type;
      break;
    }
  }

  DLOG(INFO) << "Read " << ret.size() << " image poses from " << pose_dirname;
  return ret;
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

  std::vector<xcolor::MatchTrack> match_tracks;
  std::unordered_map<colmap::image_t, colmap::Image> images;
  std::unordered_map<colmap::camera_t, colmap::Camera> cameras;

  if (FLAGS_pose_type == 2) {
    try {
      std::unordered_map<std::string, colmap::Rigid3d> image_to_pose = ReadImagePoses(FLAGS_pose_type, FLAGS_initial_pose_dirname);

      CHECK(boost::filesystem::is_regular_file(FLAGS_calibration_filename));

      YAML::Node config = YAML::LoadFile(FLAGS_calibration_filename);

      YAML::Node left_cam = config["intrinsic"]["fisheye_left"];
      double left_a11     = left_cam["projection_parameters"]["A11"].as<double>();
      double left_a22     = left_cam["projection_parameters"]["A22"].as<double>();
      int left_width      = 0;
      int left_height     = 0;
      for (const auto &[image_name, pose] : image_to_pose) {
        if (image_name.find("left") != std::string::npos) {
          auto img = cv::imread(FLAGS_images_path + "/" + image_name);
          if (img.empty()) {
            DLOG(INFO) << "Read width & height from " + FLAGS_images_path + "/" + image_name << " failed.";
          } else {
            left_width  = img.cols;
            left_height = img.rows;
            break;
          }
        }
      }

      YAML::Node right_cam = config["intrinsic"]["fisheye_right"];
      double right_a11     = right_cam["projection_parameters"]["A11"].as<double>();
      double right_a22     = right_cam["projection_parameters"]["A22"].as<double>();
      int right_width      = 0;
      int right_height     = 0;
      for (const auto &[image_name, pose] : image_to_pose) {
        if (image_name.find("right") != std::string::npos) {
          auto img = cv::imread(FLAGS_images_path + "/" + image_name);
          if (img.empty()) {
            DLOG(INFO) << "Read width & height from " + FLAGS_images_path + "/" + image_name << " failed.";
          } else {
            right_width  = img.cols;
            right_height = img.rows;
            break;
          }
        }
      }

      DLOG(INFO) << "left camera: " << left_a11 << ", " << left_a22 << ", " << left_width << ", " << left_height;
      DLOG(INFO) << "right camera: " << right_a11 << ", " << right_a22 << ", " << right_width << ", " << right_height;

      // write into image-poses.txt and camera-params.txt
      std::ofstream image_poses_file(FLAGS_output_path + "/image-poses.txt");
      image_poses_file << "camera_id image_name x y z rw rx ry rz" << std::endl;
      for (const auto &[image_name, pose] : image_to_pose) {
        if (image_name.find("left") != std::string::npos) {
          image_poses_file << "1 " + image_name + " ";
        } else {
          image_poses_file << "2 " + image_name + " ";
        }
        auto pose_c2w = colmap::Inverse(pose);
        image_poses_file << pose_c2w.translation.x() << " " << pose_c2w.translation.y() << " " << pose_c2w.translation.z() << " "
                         << pose_c2w.rotation.w() << " " << pose_c2w.rotation.x() << " " << pose_c2w.rotation.y() << " " << pose_c2w.rotation.z()
                         << std::endl;
      }
      image_poses_file.close();

      std::ofstream camera_params_file(FLAGS_output_path + "/camera-params.txt");
      camera_params_file << "camera_id model_id fx fy cx cy params..." << std::endl;
      camera_params_file << "1 4 " << left_a11 << " " << left_a22 << " " << left_width / 2 << " " << left_height / 2 << " 0 0 0 0" << std::endl;
      camera_params_file << "2 4 " << right_a11 << " " << right_a22 << " " << right_width / 2 << " " << right_height / 2 << " 0 0 0 0" << std::endl;
      camera_params_file.close();

      DLOG(INFO) << "done.";
      std::cout << "done." << std::endl;
      return 0;
    } catch (const YAML::Exception &e) {
      DLOG(ERROR) << "YAML error: " << e.what();
      return 1;
    } catch (const std::exception &e) {
      DLOG(ERROR) << "Error: " << e.what();
      return 1;
    }
  }

  // reload images
  {
    std::unordered_map<std::string, colmap::Rigid3d> image_to_pose = ReadImagePoses(FLAGS_pose_type, FLAGS_initial_pose_dirname);

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

    match_tracks = GenerateMatchPairs(corr_graph, images, config);
    DLOG(INFO) << "Loading " << match_tracks.size() << " image pairs from " << FLAGS_database_filename;
  }

  RunSFM(config, match_tracks, FLAGS_point_cloud_filename, images, cameras);

  DLOG(INFO) << "done.";
  std::cout << "done." << std::endl;

  google::ShutdownGoogleLogging();
  return 0;
}