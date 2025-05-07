
#include "xsfm_lib.h"

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
#include "migration/proto_io.h"
#include "migration/utils.h"

namespace {

std::vector<std::vector<int>> ClusterByBFS(const std::vector<MatchTrack> &match_tracks) {
  int N = int(match_tracks.size());
  std::unordered_map<Point2DInfo::Ptr, std::vector<int>> pt2pairs;

  std::vector<bool> valid(N, false);
  for (int i = 0; i < N; ++i) {
    if (!match_tracks[i].point3D.valid) continue;
    valid[i] = true;
    for (auto &pt : match_tracks[i].point2D_on_imageN) {
      pt2pairs[pt].push_back(i);
    }
  }

  std::vector<bool> seen(N, false);
  std::vector<std::vector<int>> clusters;

  for (int i = 0; i < N; ++i) {
    if (seen[i] || !valid[i]) continue;
    std::vector<int> cluster;
    std::queue<int> q;
    q.push(i);
    seen[i] = true;

    while (!q.empty()) {
      int u = q.front();
      q.pop();
      cluster.push_back(u);

      for (auto &pt : match_tracks[u].point2D_on_imageN) {
        for (int v : pt2pairs[pt]) {
          if (!seen[v] && valid[v]) {
            seen[v] = true;
            q.push(v);
          }
        }
        pt2pairs[pt].clear();
      }
    }

    clusters.push_back(std::move(cluster));
  }

  return clusters;
}

}  // namespace

std::vector<MatchTrack> GenerateMatchPairs(const colmap::CorrespondenceGraph &corr_graph,
                                           const std::unordered_map<colmap::image_t, colmap::Image> &images, const SfmConfig &config) {
  std::vector<MatchTrack> match_tracks;
  std::unordered_map<std::pair<colmap::image_t, colmap::point2D_t>, Point2DInfo::Ptr> map;
  for (auto &[image_pair_id, _] : corr_graph.NumCorrespondencesBetweenImages()) {
    std::pair<colmap::image_t, colmap::image_t> pair = colmap::Database::PairIdToImagePair(image_pair_id);

    auto image1_id        = pair.first;
    auto image2_id        = pair.second;
    auto &image1_points2D = images.at(image1_id).Points2D();
    auto &image2_points2D = images.at(image2_id).Points2D();

    auto matches = corr_graph.FindCorrespondencesBetweenImages(image1_id, image2_id);
    for (auto &match : matches) {
      MatchTrack mp;

      if (map.find(std::pair{image1_id, match.point2D_idx1}) != map.end()) {
        mp.point2D_on_imageN.emplace_back(map.at(std::pair{image1_id, match.point2D_idx1}));
      } else {
        auto &point_on_image0 = mp.point2D_on_imageN.emplace_back(std::make_shared<Point2DInfo>());

        point_on_image0->image_id    = image1_id;
        point_on_image0->camera_id   = images.at(image1_id).CameraId();
        point_on_image0->point_pixel = image1_points2D.at(match.point2D_idx1).xy;

        map[std::pair{image1_id, match.point2D_idx1}] = point_on_image0;
      }

      if (map.find(std::pair{image2_id, match.point2D_idx2}) != map.end()) {
        mp.point2D_on_imageN.emplace_back(map.at(std::pair{image2_id, match.point2D_idx2}));
      } else {
        auto &point_on_image1 = mp.point2D_on_imageN.emplace_back(std::make_shared<Point2DInfo>());

        point_on_image1->image_id    = image2_id;
        point_on_image1->camera_id   = images.at(image2_id).CameraId();
        point_on_image1->point_pixel = image2_points2D.at(match.point2D_idx2).xy;

        map[std::pair{image2_id, match.point2D_idx2}] = point_on_image1;
      }

      match_tracks.push_back(mp);
    }
  }
  DLOG(INFO) << "Load " << match_tracks.size() << " match pairs.";
  return match_tracks;
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

Eigen::Vector2d ComputePixelError(Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, const colmap::Image &image,
                                  const colmap::Camera &camera) {
  CHECK(image.HasPose());

  const double *cam_from_world_rotation    = image.CamFromWorld().rotation.coeffs().data();
  const double *cam_from_world_translation = image.CamFromWorld().translation.data();
  const double *camera_params              = camera.params.data();

  auto cost_functor          = colmap::CreateCameraCostFunction<colmap::ReprojErrorCostFunctor>(camera.model_id, point2D);
  const double *parameters[] = {cam_from_world_rotation, cam_from_world_translation, point3D.data(), camera_params};
  Eigen::Vector2d residuals;
  cost_functor->Evaluate(parameters, residuals.data(), nullptr);

  delete cost_functor;  // todo kk check memory leak

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

void PrintResidualHistogram(double threshold, ceres::Problem &problem, const std::vector<ceres::ResidualBlockId> &residual_block_ids,
                            const std::string &name) {
  ceres::Problem::EvaluateOptions evaluate_options;
  evaluate_options.apply_loss_function = false;
  evaluate_options.residual_blocks     = residual_block_ids;
  double cost;
  std::vector<double> residuals;
  problem.Evaluate(evaluate_options, &cost, &residuals, nullptr, nullptr);

  Histogram hist;
  for (auto &residual : residuals) {
    residual = std::clamp(residual, -threshold, threshold);
    hist.Add(residual);
  }
  DLOG(INFO) << name << ": " << hist.ToString(10);
}

colmap::Rigid3d FromProto(const proto::PoseMsg &pose_msg) {
  Eigen::Quaterniond rot(pose_msg.rw(), pose_msg.rx(), pose_msg.ry(), pose_msg.rz());
  Eigen::Vector3d pos(pose_msg.tx(), pose_msg.ty(), pose_msg.tz());
  return colmap::Rigid3d(rot, pos);
}

std::vector<double> ComputeRMSEByClusterCentroid(const std::vector<std::vector<int>> &clusters, const std::vector<MatchTrack> &match_tracks) {
  std::vector<double> rmse_list;

  for (const auto &cluster : clusters) {
    CHECK_GT(cluster.size(), 0);

    std::vector<Eigen::Vector3d> points;

    for (int idx : cluster) {
      auto result = match_tracks[idx];
      if (result.point3D.valid) {
        points.push_back(result.point3D.point3D);
      }
    }

    Eigen::Vector3d center(0, 0, 0);
    for (const auto &pt : points) {
      center += pt;
    }
    center /= points.size();

    double total_sq_error = 0.0;
    for (const auto &pt : points) {
      double error = (pt - center).squaredNorm();
      total_sq_error += error;
    }

    double mean_sq_error = total_sq_error / points.size();
    double rmse          = std::sqrt(mean_sq_error);
    rmse_list.push_back(rmse);
  }

  return rmse_list;
}

std::vector<std::vector<int>> DBSCANClusterIndices(const std::vector<MatchTrack> &match_tracks_total, const std::vector<int> &indices, double eps,
                                                   int min_pts) {
  std::vector<std::vector<int>> clusters;
  std::vector<bool> visited(match_tracks_total.size(), false);

  auto distance = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return (a - b).norm(); };

  for (size_t idx_i = 0; idx_i < indices.size(); ++idx_i) {
    int i = indices[idx_i];
    if (!match_tracks_total[i].point3D.valid || visited[i]) continue;

    visited[i] = true;
    std::vector<int> neighbors;

    for (int j : indices) {
      if (!match_tracks_total[j].point3D.valid) continue;
      if (distance(match_tracks_total[i].point3D.point3D, match_tracks_total[j].point3D.point3D) < eps) {
        neighbors.push_back(j);
      }
    }

    if (neighbors.size() < min_pts) continue;

    std::vector<int> cluster;
    cluster.push_back(i);

    std::deque<int> search_queue(neighbors.begin(), neighbors.end());

    while (!search_queue.empty()) {
      int j = search_queue.front();
      search_queue.pop_front();

      if (!visited[j]) {
        visited[j] = true;

        std::vector<int> sub_neighbors;
        for (int k : indices) {
          if (!match_tracks_total[k].point3D.valid) continue;
          if (distance(match_tracks_total[j].point3D.point3D, match_tracks_total[k].point3D.point3D) < eps) {
            sub_neighbors.push_back(k);
          }
        }

        if (sub_neighbors.size() >= min_pts) {
          for (int n : sub_neighbors) {
            if (std::find(neighbors.begin(), neighbors.end(), n) == neighbors.end()) {
              search_queue.push_back(n);
              neighbors.push_back(n);
            }
          }
        }
      }

      cluster.push_back(j);
    }

    clusters.push_back(cluster);
  }

  return clusters;
}

void PrintClusterMetrics(std::vector<MatchTrack> &match_tracks_valid, std::vector<std::vector<int>> &clusters,
                         std::vector<std::vector<std::vector<int>>> &sub_clusters) {
  std::vector<double> cluster_rmses = ComputeRMSEByClusterCentroid(clusters, match_tracks_valid);
  Histogram hist;
  for (int i = 0; i < clusters.size(); ++i) {
    // only use cluster with more than 1 match track
    if (clusters[i].size() > 1) {
      hist.Add(cluster_rmses[i]);
    }
  }
  DLOG(INFO) << "Cluster RMSE: " << hist.ToString(10);

  Histogram hist_count;
  for (int i = 0; i < clusters.size(); ++i) {
    // only use cluster with more than 1 match track
    if (clusters[i].size() > 1) {
      hist_count.Add(sub_clusters[i].size());
    }
  }
  DLOG(INFO) << "Cluster count: " << hist_count.ToString(20);
}

std::vector<MatchTrack> MergeMatchTracks(const std::vector<MatchTrack> &match_tracks,
                                         const std::vector<std::vector<std::vector<int>>> &sub_clusters) {
  std::vector<MatchTrack> match_tracks_ret;
  for (auto &sub_cluster : sub_clusters) {
    for (auto &cluster : sub_cluster) {
      std::unordered_set<Point2DInfo::Ptr> points2D;
      for (auto &idx : cluster) {
        for (auto &pt : match_tracks[idx].point2D_on_imageN) {
          points2D.insert(pt);
        }
      }

      MatchTrack mt_new;
      for (auto &point2D : points2D) {
        mt_new.point2D_on_imageN.push_back(point2D);
      }
      mt_new.point3D = match_tracks[cluster[0]].point3D;

      match_tracks_ret.push_back(mt_new);
    }
  }
  return match_tracks_ret;
}

bool MergeTrack(const std::vector<MatchTrack> &match_tracks, std::vector<MatchTrack> &match_tracks_merged) {
  std::vector<MatchTrack> match_tracks_valid;

  std::copy_if(match_tracks.begin(), match_tracks.end(), std::back_inserter(match_tracks_valid),
               [](const MatchTrack &mt) { return mt.point3D.valid; });

  std::vector<std::vector<int>> clusters = ClusterByBFS(match_tracks_valid);
  DLOG(INFO) << "Cluster: " << match_tracks_valid.size() << " match pairs -> " << clusters.size() << " match tracks.";

  std::vector<std::vector<std::vector<int>>> sub_clusters(clusters.size());
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
    sub_clusters[i] = DBSCANClusterIndices(match_tracks_valid, clusters[i], 0.03, 2);
  }

  PrintClusterMetrics(match_tracks_valid, clusters, sub_clusters);

  match_tracks_merged = MergeMatchTracks(match_tracks_valid, sub_clusters);
  DLOG(INFO) << "Finally, merge " << match_tracks_valid.size() << " match track into " << match_tracks_merged.size() << " tracks ";

  return true;
}
