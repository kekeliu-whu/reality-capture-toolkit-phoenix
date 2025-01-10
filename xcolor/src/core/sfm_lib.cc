
#include "sfm_lib.h"

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
#include <boost/filesystem.hpp>

#include "common/histogram.h"
#include "io/read_write.h"
#include "migration/sensor_io.h"
#include "migration/utils.h"

// merge track is impossible becuase point match pair is not a one-to-one map
void PreMergeTrack(const colmap::CorrespondenceGraph &corr_graph) {
  std::unordered_map<std::pair<colmap::image_t, colmap::point2D_t>, std::shared_ptr<colmap::Point3D>> map;
  int match_num = 0;
  for (auto &[image_pair_id, _] : corr_graph.NumCorrespondencesBetweenImages()) {
    std::pair<colmap::image_t, colmap::image_t> pair = colmap::Database::PairIdToImagePair(image_pair_id);

    auto image1_id = pair.first;
    auto image2_id = pair.second;

    auto matches = corr_graph.FindCorrespondencesBetweenImages(image1_id, image2_id);
    for (auto &match : matches) {
      auto it_point1 = map.find({image1_id, match.point2D_idx1});
      auto it_point2 = map.find({image2_id, match.point2D_idx2});

      if (it_point1 == map.end() && it_point2 == map.end()) {
        // create a new point3D
        auto point3D_new = std::make_shared<colmap::Point3D>();
        point3D_new->track.AddElement(image1_id, match.point2D_idx1);
        point3D_new->track.AddElement(image2_id, match.point2D_idx2);
        map[{image1_id, match.point2D_idx1}] = point3D_new;
        map[{image2_id, match.point2D_idx2}] = point3D_new;
      } else if (it_point1 == map.end() && it_point2 != map.end()) {
        auto point3D_old = it_point2->second;
        point3D_old->track.AddElement(image1_id, match.point2D_idx1);
        map[{image1_id, match.point2D_idx1}] = point3D_old;
      } else if (it_point1 != map.end() && it_point2 == map.end()) {
        auto point3D_old = it_point1->second;
        point3D_old->track.AddElement(image2_id, match.point2D_idx2);
        map[{image2_id, match.point2D_idx2}] = point3D_old;
      } else {
        if (it_point1->second != it_point2->second) {
          auto union_point3D = it_point1->second;
          // merge two point3Ds into one
          auto elements = it_point2->second->track.Elements();
          for (auto &element : elements) {
            union_point3D->track.AddElement(element);
            map[{element.image_id, element.point2D_idx}] = union_point3D;
          }
        }
      }
    }

    match_num += matches.size();
  }

  int point3D_num_mixed_track = 0;
  for (auto &[_, point3D] : map) {
    auto elements = point3D->track.Elements();
    std::set<colmap::image_t> images;
    for (auto &element : elements) {
      images.insert(element.image_id);
    }
    if (images.size() != elements.size()) {
      point3D_num_mixed_track++;
    }
  }

  LOG(INFO) << "Number of matches: " << match_num;
  LOG(INFO) << "Number of tracks: " << map.size();
  LOG(INFO) << "Number of mixed tracks: " << point3D_num_mixed_track;
  LOG(INFO) << "Rate of mixed tracks: " << static_cast<double>(point3D_num_mixed_track) / map.size() * 100.0 << "%";
}

std::vector<MatchPair> GenerateMatchPairs(const colmap::CorrespondenceGraph &corr_graph,
                                          const std::unordered_map<colmap::image_t, colmap::Image> &images, const SfmConfig &config) {
  std::vector<MatchPair> match_pairs;
  for (auto &[image_pair_id, _] : corr_graph.NumCorrespondencesBetweenImages()) {
    std::pair<colmap::image_t, colmap::image_t> pair = colmap::Database::PairIdToImagePair(image_pair_id);

    auto image1_id        = pair.first;
    auto image2_id        = pair.second;
    auto &image1_points2D = images.at(image1_id).Points2D();
    auto &image2_points2D = images.at(image2_id).Points2D();

    auto matches = corr_graph.FindCorrespondencesBetweenImages(image1_id, image2_id);
    for (auto &match : matches) {
      MatchPair mp;
      mp.point_on_image1.image_id    = image1_id;
      mp.point_on_image1.camera_id   = images.at(image1_id).CameraId();
      mp.point_on_image1.point_pixel = image1_points2D.at(match.point2D_idx1).xy;
      mp.point_on_image2.image_id    = image2_id;
      mp.point_on_image2.camera_id   = images.at(image2_id).CameraId();
      mp.point_on_image2.point_pixel = image2_points2D.at(match.point2D_idx2).xy;
      match_pairs.push_back(mp);
    }
  }
  LOG(INFO) << "Load " << match_pairs.size() << " match pairs.";
  return match_pairs;
}

void ParameterizeCameras(const SfmConfig &config, ceres::Problem &problem, std::unordered_map<colmap::camera_t, colmap::Camera> &cameras) {
  const bool constant_camera = !config.refine_focal_length && !config.refine_principal_point && !config.refine_extra_params;
  for (auto &[_, camera] : cameras) {
    if (constant_camera) {
      problem.SetParameterBlockConstant(camera.params.data());
      continue;
    } else {
      std::vector<int> const_camera_params;

      if (!config.refine_focal_length) {
        const colmap::span<const size_t> params_idxs = camera.FocalLengthIdxs();
        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!config.refine_principal_point) {
        const colmap::span<const size_t> params_idxs = camera.PrincipalPointIdxs();
        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }
      if (!config.refine_extra_params) {
        const colmap::span<const size_t> params_idxs = camera.ExtraParamsIdxs();
        const_camera_params.insert(const_camera_params.end(), params_idxs.begin(), params_idxs.end());
      }

      if (const_camera_params.size() > 0) {
        problem.SetManifold(camera.params.data(), new ceres::SubsetManifold(camera.params.size(), const_camera_params));
      }
    }
  }
}

void ParameterizePoses(const SfmConfig &config, ceres::Problem &problem, std::unordered_map<colmap::camera_t, colmap::Image> &images) {
  for (auto &image : images) {
    problem.SetManifold(image.second.CamFromWorld().rotation.coeffs().data(), new ceres::EigenQuaternionManifold());
  }
}

void AddReprojectFactorToProblem(ceres::Problem &problem, Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, colmap::Image &image,
                                 colmap::Camera &camera, ceres::LossFunction *loss_function,
                                 std::vector<ceres::ResidualBlockId> &residual_block_ids) {
  CHECK(image.HasPose());
  image.CamFromWorld().rotation.normalize();

  double *cam_from_world_rotation    = image.CamFromWorld().rotation.coeffs().data();
  double *cam_from_world_translation = image.CamFromWorld().translation.data();
  double *camera_params              = camera.params.data();

  auto residual_block_id =
      problem.AddResidualBlock(colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(camera.model_id, point2D), loss_function,
                               cam_from_world_rotation, cam_from_world_translation, point3D.data(), camera_params);
  problem.SetManifold(cam_from_world_rotation, new ceres::EigenQuaternionManifold());

  residual_block_ids.push_back(residual_block_id);
}

Eigen::Vector2d ComputePixelError(Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, colmap::Image &image, colmap::Camera &camera) {
  CHECK(image.HasPose());
  image.CamFromWorld().rotation.normalize();

  double *cam_from_world_rotation    = image.CamFromWorld().rotation.coeffs().data();
  double *cam_from_world_translation = image.CamFromWorld().translation.data();
  double *camera_params              = camera.params.data();

  auto cost_functor    = colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(camera.model_id, point2D);
  double *parameters[] = {cam_from_world_rotation, cam_from_world_translation, point3D.data(), camera_params};
  Eigen::Vector2d residuals;
  cost_functor->Evaluate(parameters, residuals.data(), nullptr);

  delete cost_functor; // todo kk check memory leak

  return residuals;
}

double ComputeLidarError(Eigen::Vector3d &point3D, const Eigen::Vector3d &center, const Eigen::Vector3d &normal) {
  return (point3D - center).dot(normal);
}

void AddLidarFactorToProblem(ceres::Problem &problem, Eigen::Vector3d &point3D, const Eigen::Vector3d &center, const Eigen::Vector3d &normal,
                             ceres::LossFunction *loss_function, std::vector<ceres::ResidualBlockId> &residual_block_ids) {
  auto residual_block_id = problem.AddResidualBlock(LidarPlaneCostFunction::Create(center, normal), loss_function, point3D.data());

  residual_block_ids.push_back(residual_block_id);
}

void AddPosePriorsToProblem(const SfmConfig &config, ceres::Problem &problem, const std::unordered_map<colmap::image_t, colmap::Rigid3d> &pose_priors,
                            std::unordered_set<colmap::image_t> &optimized_image_ids, std::unordered_map<colmap::image_t, colmap::Image> &images) {
  Eigen::DiagonalMatrix<double, 6> info_matrix;
  info_matrix.diagonal() << 1 / config.pose_prior_translation_weight, 1 / config.pose_prior_translation_weight,
      1 / config.pose_prior_translation_weight, 1 / config.pose_prior_rotation_weight, 1 / config.pose_prior_rotation_weight,
      1 / config.pose_prior_rotation_weight;

  for (auto &image_id : optimized_image_ids) {
    auto &pose       = images.at(image_id).CamFromWorld();
    auto &pose_prior = pose_priors.at(image_id);
    problem.AddResidualBlock(PosePriorCostFunction::Create(pose_prior.rotation, pose_prior.translation, info_matrix),
                             new ceres::ScaledLoss(nullptr, config.pose_prior_scale_weight, ceres::DO_NOT_TAKE_OWNERSHIP),
                             pose.rotation.coeffs().data(), pose.translation.data());
    problem.SetManifold(pose.rotation.coeffs().data(), new ceres::EigenQuaternionManifold());
  }
}

void PrintResidualHistogram(ceres::Problem &problem, const std::vector<ceres::ResidualBlockId> &residual_block_ids, const std::string &name) {
  ceres::Problem::EvaluateOptions evaluate_options;
  evaluate_options.apply_loss_function = false;
  evaluate_options.residual_blocks     = residual_block_ids;
  double cost;
  std::vector<double> residuals;
  problem.Evaluate(evaluate_options, &cost, &residuals, nullptr, nullptr);

  Histogram hist;
  for (auto &residual : residuals) {
    hist.Add(residual);
  }
  LOG(INFO) << name << ": " << hist.ToString(10);
}

colmap::Rigid3d FromProto(const PoseMsg &pose_msg) {
  Eigen::Quaterniond rot(pose_msg.rw(), pose_msg.rx(), pose_msg.ry(), pose_msg.rz());
  Eigen::Vector3d pos(pose_msg.tx(), pose_msg.ty(), pose_msg.tz());
  return colmap::Rigid3d(rot, pos);
}
