
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
#include <boost/filesystem.hpp>

#include "common/histogram.h"
#include "core/sfm_lib.h"
#include "io/read_write.h"
#include "migration/sensor_io.h"
#include "migration/utils.h"

DEFINE_string(image_path, "D:/BaiduNetdiskDownload/s10-colmap/images", "SFM databaset filename");
DEFINE_string(point_cloud_filename, "D:/BaiduNetdiskDownload/s10-colmap/normals-downsampled.ply", "Point cloud filename");
DEFINE_string(output_path, "D:/BaiduNetdiskDownload/s10-colmap", "Output path");
DEFINE_string(database_filename, "D:/BaiduNetdiskDownload/s10-colmap/test.db", "Database filename");
DEFINE_string(initial_pose_filename, "D:/BaiduNetdiskDownload/s10-colmap/transforms.json", "Initial pose filename");

void SaveTriangulatedPoints(const std::vector<MatchResult> &match_results, const std::string &filename) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud_out;
  for (int i = 0; i < match_results.size(); ++i) {
    if (match_results[i].valid) {
      auto point3D = match_results[i].point3D;
      pcl::PointXYZINormal np;
      np.x = point3D.x();
      np.y = point3D.y();
      np.z = point3D.z();
      point_cloud_out.push_back(np);
    }
  }
  pcl::io::savePCDFileBinary(filename, point_cloud_out);
}

void SaveImagePoses(const std::string &filename, const std::unordered_set<colmap::image_t> &optimized_image_ids,
                    const std::unordered_map<colmap::image_t, colmap::Image> &images,
                    const std::unordered_map<colmap::camera_t, colmap::Rigid3d> &pose_priors) {
  LOG(INFO) << "Saving image information into " << FLAGS_output_path + "/images.bin";
  std::vector<colmap::Image> images_out;
  for (auto &image_id : optimized_image_ids) {
    images_out.push_back(images.at(image_id));
  }
  WriteImagesBinary(FLAGS_output_path + "/images.bin", images_out);

  LOG(INFO) << "Saving image poses into " << filename;
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
  std::vector<colmap::Camera> cameras_out;
  for (auto &[image_id, _] : cameras) {
    cameras_out.push_back(cameras.at(image_id));
  }
  WriteCamerasBinary(FLAGS_output_path + "/cameras.bin", cameras_out);

  std::ofstream infile(filename);
  infile << "camera_id model_id image_name x y z rw rx ry rz" << std::endl;
  for (auto &e : cameras) {
    infile << std::fixed << std::setprecision(6) << e.first << " " << (int)e.second.model_id << " " << e.second.params[0] << " " << e.second.params[1]
           << " " << e.second.params[2] << " " << e.second.params[3] << " " << e.second.params[4] << " " << e.second.params[5] << " "
           << e.second.params[6] << " " << e.second.params[7] << std::endl;
  }
}

void RunSFM(const SfmConfig &config, const std::vector<MatchPair> &match_pairs, const std::string &point_cloud_filename,
            std::unordered_map<colmap::image_t, colmap::Image> &images, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud;
  int load_ply_status = pcl::io::loadPLYFile(point_cloud_filename, point_cloud);
  CHECK_NE(load_ply_status, -1);
  LOG(INFO) << "Load " << point_cloud.size() << " points.";

  // build kdtree
  pcl::KdTreeFLANN<pcl::PointXYZINormal> kdtree;
  kdtree.setInputCloud(point_cloud.makeShared());

  std::unordered_map<colmap::image_t, colmap::Rigid3d> pose_priors;
  for (auto &[image_id, image] : images) {
    pose_priors[image.ImageId()] = image.CamFromWorld();
  }

  for (int iter = 0; iter < config.outer_opt_num; ++iter) {
    ceres::Problem::Options problem_options;
    problem_options.loss_function_ownership = ceres::DO_NOT_TAKE_OWNERSHIP;
    ceres::Problem problem{problem_options};
    auto loss_function_image = std::make_shared<ceres::CauchyLoss>(config.reproject_cauchy_loss_scale);
    auto loss_function_lidar = std::make_shared<ceres::ScaledLoss>(new ceres::CauchyLoss(config.lidar_cauchy_loss_scale), config.lidar_loss_scale,
                                                                   ceres::DO_NOT_TAKE_OWNERSHIP);
    std::vector<ceres::ResidualBlockId> residual_block_ids;
    std::vector<ceres::ResidualBlockId> lidar_residual_block_ids;

    std::vector<MatchResult> match_results;
    match_results.resize(match_pairs.size());

    std::unordered_set<colmap::image_t> optimized_image_ids;

#pragma omp parallel for
    for (int i = 0; i < match_pairs.size(); ++i) {
      auto &match_pair      = match_pairs[i];
      auto &point_on_image1 = match_pair.point_on_image1;
      auto &point_in_image2 = match_pair.point_on_image2;
      auto &image1          = images[point_on_image1.image_id];
      auto &image2          = images[point_in_image2.image_id];
      auto &camera1         = cameras[point_on_image1.camera_id];
      auto &camera2         = cameras[point_in_image2.camera_id];
      auto &point3D         = match_results[i].point3D;

      CHECK(image1.HasPose()) << "Image " << point_on_image1.image_id << " has no pose.";
      CHECK(image2.HasPose()) << "Image " << point_in_image2.image_id << " has no pose.";

      auto cam_from_world1 = image1.CamFromWorld().ToMatrix();
      auto cam_from_world2 = image2.CamFromWorld().ToMatrix();

      bool ok = colmap::TriangulatePoint(cam_from_world1, cam_from_world2, camera1.CamFromImg(point_on_image1.point_pixel),
                                         camera2.CamFromImg(point_in_image2.point_pixel), &point3D);

      if (!ok) {
        continue;
      }

      Eigen::Vector3d cam1_center = -(cam_from_world1.block<3, 3>(0, 0).transpose() * cam_from_world1.block<3, 1>(0, 3));
      Eigen::Vector3d cam2_center = -(cam_from_world2.block<3, 3>(0, 0).transpose() * cam_from_world2.block<3, 1>(0, 3));

      double angle = colmap::CalculateTriangulationAngle(cam1_center, cam2_center, point3D);
      if (std::abs(angle) < config.min_tri_angle * M_PI / 180) {
        continue;
      }

      std::vector<float> distances;
      std::vector<int> indices;
      kdtree.nearestKSearch(pcl::PointXYZINormal{(float)point3D.x(), (float)point3D.y(), (float)point3D.z()}, 1, indices, distances);

      auto nearest_point = point_cloud[indices[0]];

      auto projection_error1 = ComputePixelError(point3D, point_on_image1.point_pixel, image1, camera1);
      auto projection_error2 = ComputePixelError(point3D, point_in_image2.point_pixel, image2, camera2);
      if (projection_error1.norm() > config.reproject_error_outlier_thresholds[iter] ||
          projection_error2.norm() > config.reproject_error_outlier_thresholds[iter]) {
        continue;
      }

      auto lidar_error =
          ComputeLidarError(point3D, nearest_point.getVector3fMap().cast<double>(), nearest_point.getNormalVector3fMap().cast<double>());
      if (std::abs(lidar_error) > config.lidar_error_outlier_thresholds[iter]) {
        continue;
      }

      match_results[i].valid = 1;

#pragma omp critical
      {
        optimized_image_ids.insert(point_on_image1.image_id);
        optimized_image_ids.insert(point_in_image2.image_id);
        LOG_EVERY_N(INFO, 10000) << "Triangulate point " << i;
        AddReprojectFactorToProblem(problem, point3D, point_on_image1.point_pixel, image1, camera1, loss_function_image.get(), residual_block_ids);
        AddReprojectFactorToProblem(problem, point3D, point_in_image2.point_pixel, image2, camera2, loss_function_image.get(), residual_block_ids);
        AddLidarFactorToProblem(problem, point3D, nearest_point.getVector3fMap().cast<double>(), nearest_point.getNormalVector3fMap().cast<double>(),
                                loss_function_lidar.get(), lidar_residual_block_ids);
      }
    }
    LOG(INFO) << optimized_image_ids.size() << " images are used.";
    LOG(INFO) << images.size() << " images in total.";
    ParameterizeCameras(config, problem, cameras);

    AddPosePriorsToProblem(config, problem, pose_priors, optimized_image_ids, images);

    int valid_count = std::accumulate(match_results.begin(), match_results.end(), 0, [](int sum, auto &a) { return sum + a.valid; });
    LOG(INFO) << valid_count << "/" << match_pairs.size() << " (" << valid_count * 100.0 / match_pairs.size() << "%) matches are valid.";

    PrintResidualHistogram(problem, residual_block_ids, "image");
    PrintResidualHistogram(problem, lidar_residual_block_ids, "lidar");

    SaveTriangulatedPoints(match_results, FLAGS_output_path + "/triangulated" + std::to_string(iter) + ".pcd");

    ceres::Solver::Summary summary;
    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.num_threads                  = config.ba_optimization_num_threads;
    options.max_num_iterations           = 20;
    ceres::Solve(options, &problem, &summary);
    LOG(INFO) << summary.FullReport();

    SaveTriangulatedPoints(match_results, FLAGS_output_path + "/triangulated" + std::to_string(iter) + "-refined.pcd");

    SaveImagePoses(FLAGS_output_path + "/image-poses.txt", optimized_image_ids, images, pose_priors);
    SaveCameraParams(FLAGS_output_path + "/camera-params.txt", cameras);
  }
}

void GiveInitialPosesToImages(std::unordered_map<colmap::image_t, colmap::Image> &images, PoseMsgList &pose_msg_list) {
  const auto &pose_msgs = pose_msg_list.pose_msgs();
  for (auto &[_, image] : images) {
    auto filename = boost::filesystem::path(image.Name());
    filename.replace_extension();

    double timestamp = std::stod(filename.filename().string());
    auto it          = std::upper_bound(pose_msgs.begin(), pose_msgs.end(), timestamp,
                                        [](double timestamp, const PoseMsg &pose_msg) { return timestamp < pose_msg.timestamp(); });
    int idx          = std::distance(pose_msgs.begin(), it);
    if (idx >= 1 && idx < pose_msgs.size()) {
      auto &pose1   = FromProto(pose_msgs[idx - 1]);
      auto &pose2   = FromProto(pose_msgs[idx]);
      double factor = (timestamp - pose_msgs[idx - 1].timestamp()) / (pose_msgs[idx].timestamp() - pose_msgs[idx - 1].timestamp());
      CHECK_GE(factor, 0);
      CHECK_LE(factor, 1);

      colmap::Rigid3d pose_imu_to_world(pose1.rotation.slerp(factor, pose2.rotation),
                                        pose1.translation + factor * (pose2.translation - pose1.translation));

      // todo kk hard code
      colmap::Rigid3d lidar_to_imu(Eigen::Quaterniond::Identity(), Eigen::Vector3d(-0.011, -0.0234, 0.044));

      Eigen::Matrix3d rot_mat_left_cam;
      rot_mat_left_cam << 0.780897, -0.498768, 0.376072, 0.422435, -0.0218368, -0.90613, 0.460161, 0.86646, 0.193645;
      colmap::Rigid3d lidar_to_left_cam(Eigen::Quaterniond(rot_mat_left_cam).normalized(), Eigen::Vector3d(-0.105267, 0.00935615, -0.050799));

      Eigen::Matrix3d rot_mat_right_cam;
      rot_mat_right_cam << -0.751368, -0.557914, -0.35239, 0.393254, 0.0502512, -0.918056, 0.529904, -0.828376, 0.181645;
      colmap::Rigid3d lidar_to_right_cam(Eigen::Quaterniond(rot_mat_right_cam).normalized(), Eigen::Vector3d(-0.192389, 0.0510826, 0.0242671));

      if (image.Name().find("left") != std::string::npos) {
        image.SetCamFromWorld(lidar_to_left_cam * colmap::Inverse(lidar_to_imu) * colmap::Inverse(pose_imu_to_world));
      } else if (image.Name().find("right") != std::string::npos) {
        image.SetCamFromWorld(lidar_to_right_cam * colmap::Inverse(lidar_to_imu) * colmap::Inverse(pose_imu_to_world));
      } else {
        LOG(FATAL);
      }
    }
  }
  int valid_count = std::accumulate(images.begin(), images.end(), 0, [](int sum, const auto &p) { return sum + p.second.HasPose(); });
  LOG(INFO) << "Give initial poses to images done: " << valid_count << "/" << images.size();
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  LOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  SfmConfig config{cores_used};

  std::unordered_map<std::string, colmap::Rigid3d> image_to_c2w;
  std::unordered_set<std::string> image_names_with_pose;
  std::vector<colmap::Rigid3d> pose_list;
  std::vector<MatchPair> match_pairs;
  std::unordered_map<colmap::image_t, colmap::Image> images;
  std::unordered_map<colmap::camera_t, colmap::Camera> cameras;

  {
    std::ifstream file(FLAGS_initial_pose_filename);
    CHECK(file) << FLAGS_initial_pose_filename;

    std::string json_str{std::istreambuf_iterator<char>(file), std::istreambuf_iterator<char>()};

    rapidjson::Document doc;
    doc.ParseStream(rapidjson::StringStream{json_str.c_str()});

    auto &frames = doc["frames"];
    for (int i = 0; i < frames.Size(); ++i) {
      auto &frame = frames[i];

      auto &mat = frame["transform_matrix"];
      Eigen::Matrix4d T;
      T << mat[0][0].GetDouble(), mat[0][1].GetDouble(), mat[0][2].GetDouble(), mat[0][3].GetDouble(), mat[1][0].GetDouble(), mat[1][1].GetDouble(),
          mat[1][2].GetDouble(), mat[1][3].GetDouble(), mat[2][0].GetDouble(), mat[2][1].GetDouble(), mat[2][2].GetDouble(), mat[2][3].GetDouble(), 0,
          0, 0, 1;

      colmap::Rigid3d pose;
      pose.rotation = T.block<3, 3>(0, 0);
      pose.rotation.normalize();
      pose.translation = T.block<3, 1>(0, 3);

      std::string file_path = frame["file_path"].GetString();
      auto file             = boost::filesystem::path(file_path);
      auto timestamp        = frame["timestamp"].GetUint64();

      auto real_filename = file.parent_path().string() + "/" + std::to_string(timestamp) + ".png";

      image_to_c2w[real_filename] = pose;
      image_names_with_pose.insert(real_filename);
    }
  }

  // reload images
  {
    colmap::Database database(FLAGS_database_filename);
    auto database_cache    = colmap::DatabaseCache::Create(database, config.min_num_matches, config.ignore_watermarks, image_names_with_pose);
    cameras                = database_cache->Cameras();
    images                 = database_cache->Images();
    const auto &corr_graph = *database_cache->CorrespondenceGraph();

    for (auto &image : images) {
      image.second.SetCamFromWorld(colmap::Inverse(image_to_c2w[image.second.Name()]));
      Eigen::Matrix3d mat;
      mat << 1, 0, 0, 0, -1, 0, 0, 0, -1;
      image.second.CamFromWorld().translation = mat * image.second.CamFromWorld().translation;
      image.second.CamFromWorld().rotation    = Eigen::Quaterniond(mat) * image.second.CamFromWorld().rotation;
    }

    match_pairs = GenerateMatchPairs(corr_graph, images, config);
  }

  RunSFM(config, match_pairs, FLAGS_point_cloud_filename, images, cameras);

  LOG(INFO) << "done.";

  return 0;
}