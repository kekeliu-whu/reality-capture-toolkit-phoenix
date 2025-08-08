#include "io/io_utils.h"
#include "io/colmap_io.h"

void SaveTriangulatedPoints(const std::vector<xcolor::MatchTrack> &match_tracks, const std::string &filename) {
  pcl::PointCloud<pcl::PointXYZINormal> point_cloud_out;
  for (int i = 0; i < match_tracks.size(); ++i) {
    if (match_tracks[i].constraint_type != xcolor::TrackConstraintType::kUnconstrained) {
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

void SaveImagePoses(const std::string &filename_txt, const std::string &filename_bin, const std::unordered_set<colmap::image_t> &optimized_image_ids,
                    const std::unordered_map<colmap::image_t, colmap::Image> &images,
                    const std::unordered_map<colmap::camera_t, colmap::Rigid3d> &pose_priors) {
  DLOG(INFO) << "Saving image poses to " << filename_bin;
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
  xcolor::WriteImagesBinary(filename_bin, images_out);

  DLOG(INFO) << "Saving image poses to " << filename_txt;
  std::ofstream infile(filename_txt);
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

void SaveCameraParams(const std::string &filename_txt, const std::string &filename_bin,
                      const std::unordered_map<colmap::camera_t, colmap::Camera> &cameras) {
  DLOG(INFO) << "Saving camera params to " << filename_bin;
  std::vector<colmap::Camera> cameras_out;
  for (auto &[image_id, _] : cameras) {
    cameras_out.push_back(cameras.at(image_id));
  }
  xcolor::WriteCamerasBinary(filename_bin, cameras_out);

  DLOG(INFO) << "Saving camera params to " << filename_txt;
  std::ofstream infile(filename_txt);
  infile << "camera_id model_id fx fy cx cy params..." << std::endl;
  for (auto &e : cameras) {
    infile << std::fixed << std::setprecision(6) << e.first << " " << (int)e.second.model_id << " " << e.second.params[0] << " " << e.second.params[0]
           << " " << e.second.params[1] << " " << e.second.params[2] << " " << e.second.params[3] << " " << e.second.params[4] << " "
           << e.second.params[5] << " " << e.second.params[6] << std::endl;
  }
}
