#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/scene/database.h>
#include <colmap/scene/database_cache.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_types.h>
#include <boost/filesystem.hpp>
#include <nlohmann/json.hpp>

#include "core/run_ba.h"
#include "core/xsfm_lib.h"
#include "io/xml_io.h"

DEFINE_string(point_cloud_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/colorized.las_normals.pcd", "Point cloud filename");
DEFINE_string(point_cloud_offset_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/colorized.las_offset.json", "Point cloud offset filename");
DEFINE_string(database_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/xsfm.db", "Database filename");
DEFINE_string(initial_pose_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/images/ImgPose.txt", "Initial pose filename");
DEFINE_string(images_path, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/images", "");

DEFINE_string(output_path, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm", "Output path");

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

  RunTwoViewBA(config, FLAGS_output_path, kdtree, point_cloud, pose_priors, images, cameras, match_tracks_coarse);

  MergeTrack(match_tracks_coarse, match_tracks_fine, config.min_track_len);

  RunMultipleViewBA(config, FLAGS_output_path, kdtree, point_cloud, pose_priors, images, cameras, match_tracks_fine);
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

void ReadPointCloudOffset(const std::string &filename, Eigen::Vector2d &offset, std::string &proj_str) {
  std::ifstream infile(filename);
  if (infile.is_open()) {
    nlohmann::json j;
    infile >> j;
    
    offset.x() = j["offset_x"];
    offset.y() = j["offset_y"];
    proj_str = j["proj4_string"];
    
    DLOG(INFO) << "Read point cloud offset: " << offset.transpose() << ", proj_str: " << proj_str;
  } else {
    DLOG(FATAL) << "Failed to open file: " << filename;
  }
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
  Eigen::Vector2d offset;
  std::string proj_str;
  {
    ReadPointCloudOffset(FLAGS_point_cloud_offset_filename, offset, proj_str);
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
  xcolor::SaveXml(FLAGS_output_path + "/mvs.xml", images, cameras, match_tracks_fine, offset, proj_str, FLAGS_images_path);

  DLOG(INFO) << "done.";
  std::cout << "done." << std::endl;

  google::ShutdownGoogleLogging();
  return 0;
}