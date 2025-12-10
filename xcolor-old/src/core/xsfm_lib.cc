#include "xsfm_lib.h"

#include <ceres/ceres.h>
#include <colmap/estimators/cost_functions.h>
#include <colmap/scene/database.h>
#include <colmap/scene/database_cache.h>
#include <spdlog/spdlog.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>

#include "common/histogram.h"
#include "io/colmap_io.h"

namespace xcolor {

std::vector<std::vector<int>> ClusterByBFS(const std::vector<MatchTrack> &match_tracks) {
  int N = int(match_tracks.size());
  // Union-Find structure
  std::vector<int> parent(N);
  for (int i = 0; i < N; ++i) parent[i] = i;

  // Find root
  auto find = [&](int x) {
    while (parent[x] != x) {
      parent[x] = parent[parent[x]];
      x = parent[x];
    }
    return x;
  };

  // Union
  auto unite = [&](int x, int y) {
    int fx = find(x), fy = find(y);
    if (fx != fy) parent[fx] = fy;
  };

  // Build point to track mapping
  std::unordered_map<Point2DInfo::Ptr, int> pt2track;
  for (int i = 0; i < N; ++i) {
    for (auto &pt : match_tracks[i].point2D_on_imageN) {
      auto it = pt2track.find(pt);
      if (it != pt2track.end()) {
        unite(i, it->second);
      } else {
        pt2track[pt] = i;
      }
    }
  }

  // Collect members of each set
  std::unordered_map<int, std::vector<int>> clusters_map;
  for (int i = 0; i < N; ++i) {
    int root = find(i);
    clusters_map[root].push_back(i);
  }

  std::vector<std::vector<int>> clusters;
  clusters.reserve(clusters_map.size());
  for (auto &kv : clusters_map) {
    clusters.push_back(std::move(kv.second));
  }
  return clusters;
}

// Read all matching pairs from colmap into Track, each Track contains two Point2DInfo
// A single feature point has only one Point2DInfo instance, for example, the feature point (image_id, point2D_idx) corresponds to only one Point2DInfo instance (implemented via smart pointers)
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
        point_on_image0->point2D_idx = match.point2D_idx1;

        map[std::pair{image1_id, match.point2D_idx1}] = point_on_image0;
      }

      if (map.find(std::pair{image2_id, match.point2D_idx2}) != map.end()) {
        mp.point2D_on_imageN.emplace_back(map.at(std::pair{image2_id, match.point2D_idx2}));
      } else {
        auto &point_on_image1 = mp.point2D_on_imageN.emplace_back(std::make_shared<Point2DInfo>());

        point_on_image1->image_id    = image2_id;
        point_on_image1->camera_id   = images.at(image2_id).CameraId();
        point_on_image1->point_pixel = image2_points2D.at(match.point2D_idx2).xy;
        point_on_image1->point2D_idx = match.point2D_idx2;

        map[std::pair{image2_id, match.point2D_idx2}] = point_on_image1;
      }

      match_tracks.push_back(mp);
    }
  }
  spdlog::info("Load {} match pairs.", match_tracks.size());
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

void AddReprojectFactorToProblem(ceres::Problem &problem, Eigen::Vector3d &point3D, const Eigen::Vector2d &point2D, colmap::Image &image,
                                 colmap::Camera &camera, ceres::LossFunction *loss_function,
                                 std::vector<ceres::ResidualBlockId> &residual_block_ids) {
  if (!image.HasPose()) { spdlog::error("Check failed"); exit(1); }
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
  if (!image.HasPose()) { spdlog::error("Check failed"); exit(1); }

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
                             new ceres::ScaledLoss(nullptr, config.pose_prior_weight_scale, ceres::TAKE_OWNERSHIP), pose.rotation.coeffs().data(),
                             pose.translation.data());
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
  spdlog::info("{}: {}", name, hist.ToString(10));
}

std::vector<double> ComputeRMSEByClusterCentroid(const std::vector<std::vector<int>> &clusters, const std::vector<MatchTrack> &match_tracks) {
  std::vector<double> rmse_list;

  for (const auto &cluster : clusters) {
    if (cluster.size() <= 0) { spdlog::error("Check failed: cluster.size() > 0"); exit(1); }

    std::vector<Eigen::Vector3d> points;

    for (int idx : cluster) {
      points.push_back(match_tracks[idx].point3D.point3D);
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
                                                   int min_pts, int min_track_size) {
  if (min_track_size < 2) { spdlog::error("Check failed: min_track_size >= 2"); exit(1); }
  if (match_tracks_total.size() + 1 < min_track_size) {
    return {};
  }

  std::vector<std::vector<int>> clusters;
  std::vector<int8_t> visited(match_tracks_total.size(), false);

  auto distance = [](const Eigen::Vector3d &a, const Eigen::Vector3d &b) { return (a - b).norm(); };

  for (size_t idx_i = 0; idx_i < indices.size(); ++idx_i) {
    int i = indices[idx_i];
    if (visited[i]) continue;

    visited[i] = true;
    std::vector<int> neighbors;

    for (int j : indices) {
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

    if (cluster.size() >= min_track_size) {
      clusters.push_back(cluster);
    }
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
  spdlog::info("Cluster RMSE: {}", hist.ToString(10));

  Histogram hist_count;
  for (int i = 0; i < clusters.size(); ++i) {
    // only use cluster with more than 1 match track
    if (clusters[i].size() > 1) {
      hist_count.Add(sub_clusters[i].size());
    }
  }
  spdlog::info("Cluster count: {}", hist_count.ToString(20));
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

bool MergeTrack(const std::vector<MatchTrack> &match_tracks_coarse, std::vector<MatchTrack> &match_tracks_fine, int min_track_size) {
  std::vector<MatchTrack> match_tracks_valid;

  std::copy_if(match_tracks_coarse.begin(), match_tracks_coarse.end(), std::back_inserter(match_tracks_valid),
               [](const MatchTrack &mt) { return mt.constraint_type != TrackConstraintType::kUnconstrained; });

  spdlog::info("MergeTrack: {} valid match tracks.", match_tracks_valid.size());
  std::vector<std::vector<int>> clusters = ClusterByBFS(match_tracks_valid);
  spdlog::info("Cluster: {} match pairs -> {} match tracks.", match_tracks_valid.size(), clusters.size());

  int count = 0;
  std::vector<std::vector<std::vector<int>>> sub_clusters(clusters.size());
#pragma omp parallel for
  for (int i = 0; i < static_cast<int>(clusters.size()); ++i) {
    sub_clusters[i] = DBSCANClusterIndices(match_tracks_valid, clusters[i], 0.03, 2, min_track_size);
#pragma omp critical
    {
      if (++count % 10000 == 0) {
        spdlog::info("DBSCANCluster point set {}", count);
      }
    }
  }

  spdlog::info("DBSCAN clustering finished.");

  PrintClusterMetrics(match_tracks_valid, clusters, sub_clusters);

  match_tracks_fine = MergeMatchTracks(match_tracks_valid, sub_clusters);
  spdlog::info("Finally, merge {} match track into {} tracks", match_tracks_valid.size(), match_tracks_fine.size());

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

void PrintMatchTrackStatistics(const std::vector<xcolor::MatchTrack> &match_tracks) {
  int matched_visual_only_count      = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &e) {
    return sum + (e.constraint_type == xcolor::TrackConstraintType::kVisualOnly);
  });
  int matched_visual_and_lidar_count = std::accumulate(match_tracks.begin(), match_tracks.end(), 0, [](int sum, auto &e) {
    return sum + (e.constraint_type == xcolor::TrackConstraintType::kVisualAndLidar);
  });
  spdlog::info("matched_visual_only: {:.2f}% matched_visual_and_lidar_count: {:.2f}%", matched_visual_only_count * 100.0 / match_tracks.size(), matched_visual_and_lidar_count * 100.0 / match_tracks.size());
}

}  // namespace xcolor
