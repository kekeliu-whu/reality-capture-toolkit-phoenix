#define SOPHUS_DISABLE_ENSURES

#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/gicp.h>
#include <spdlog/spdlog.h>

// BTC library files are located directly in pgo/src/.
// Use btc_compat/ stubs to satisfy ros/ros.h and visualization_msgs headers.
#include "btc_compat/msvc_chrono_compat.h"
#include "BTC.h"

#include "factor/btc_factor.h"
#include "factor/gravity_factor.h"
#include "factor/local_parameterization_se3.h"
#include "factor/pose_graph_edge_factor.h"
#include "factor/rtk_factor.h"
#include "io/local_enu_transformer.h"
#include "io/read_write.h"
#include "optimizer.h"

namespace {

using PointT = pcl::PointXYZI;

double CalcFitnessScore(const pcl::PointCloud<PointT>::ConstPtr &cloud1,
                        const pcl::PointCloud<PointT>::ConstPtr &cloud2,
                        const Sophus::SE3d &relpose_2to1,
                        double max_range) {
  pcl::search::KdTree<PointT>::Ptr tree_(new pcl::search::KdTree<PointT>());
  tree_->setInputCloud(cloud1);

  double fitness_score = 0.0;

  // Transform the input dataset using the final transformation
  pcl::PointCloud<PointT> input_transformed;
  pcl::transformPointCloud(*cloud2, input_transformed,
                           relpose_2to1.matrix().cast<float>());

  std::vector<int> nn_indices(1);
  std::vector<float> nn_dists(1);

  // For each point in the source dataset
  int nr = 0;
  for (size_t i = 0; i < input_transformed.points.size(); ++i) {
    // Find its nearest neighbor in the target
    tree_->nearestKSearch(input_transformed.points[i], 1, nn_indices, nn_dists);

    // Deal with occlusions (incomplete targets)
    if (nn_dists[0] <= max_range) {
      // Add to the fitness score
      fitness_score += nn_dists[0];
      nr++;
    }
  }

  if (nr > 0)
    return (fitness_score / nr);
  else
    return (std::numeric_limits<double>::max());
}

bool MatchGICP(pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
               pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
               Sophus::SE3d &T_source_to_target,
               double gicp_fitness_score_threshold,
               int gicp_max_iterations,
               double gicp_transform_epsilon) {
  pcl::GeneralizedIterativeClosestPoint<pcl::PointXYZI, pcl::PointXYZI> gicp;
  if (!source.get()) {
    spdlog::error("Check failed");
    exit(1);
  }
  if (!target.get()) {
    spdlog::error("Check failed");
    exit(1);
  }
  gicp.setInputSource(source);
  gicp.setInputTarget(target);

  gicp.setMaximumIterations(gicp_max_iterations);
  gicp.setTransformationEpsilon(gicp_transform_epsilon);

  pcl::PointCloud<pcl::PointXYZI> aligned_source;
  gicp.align(aligned_source, T_source_to_target.matrix().cast<float>());
  T_source_to_target =
      Sophus::SE3d(gicp.getFinalTransformation().cast<double>());

  double score = CalcFitnessScore(target, source, T_source_to_target, 2.0);
  if (score < gicp_fitness_score_threshold) {
    spdlog::info("good match: {}", score);
    return true;
  }
  return false;
}

void AddParameters(ceres::Problem &problem,
                   std::vector<TimestampedPointCloud> &submaps) {
  ceres::Manifold *local_parameterization =
      LocalParameterizationSE3::Create();

  for (auto &submap : submaps) {
    problem.AddParameterBlock(submap.pose.data(), 7, local_parameterization);
  }
}

void AddGravityConstraits(ceres::Problem &problem,
                          std::vector<TimestampedPointCloud> &submaps,
                          std::set<ceres::ResidualBlockId> &prior_residual_blocks,
                          const proto::PgoConfig &config) {
  Eigen::DiagonalMatrix<double, 3> gravity_align_sqrt_information;
  gravity_align_sqrt_information.diagonal() << 1 / (config.gravity_align_rotation_error() / 180.0 * M_PI),
      1 / (config.gravity_align_rotation_error() / 180.0 * M_PI),
      1 / (config.gravity_align_rotation_error() / 180.0 * M_PI);

  for (auto &submap : submaps) {
    // todo
    auto residual_block_id = problem.AddResidualBlock(GravityAlignFactor::Create(submap.pose, Eigen::Vector3d(0, 0, -1), gravity_align_sqrt_information), nullptr, submap.pose.data());
    prior_residual_blocks.insert(residual_block_id);
  }
}

void AddAdjacentConstraints(
    ceres::Problem &problem,
    std::vector<TimestampedPointCloud> &submaps,
    std::set<ceres::ResidualBlockId> &prior_residual_blocks,
    const proto::PgoConfig &config) {
  Eigen::DiagonalMatrix<double, 6> odom_sqrt_information;
  odom_sqrt_information.diagonal() << 1 / config.odom_edge_translation_error(),
      1 / config.odom_edge_translation_error(),
      1 / config.odom_edge_translation_error(),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI);

  for (size_t i = 0; i < submaps.size() - 1; ++i) {
    auto &submap_a = submaps[i];
    auto &submap_b = submaps[i + 1];

    Sophus::SE3d T_ab      = submap_a.pose.inverse() * submap_b.pose;
    auto residual_block_id = problem.AddResidualBlock(
        PoseGraphEdgeFactor::Create(T_ab, odom_sqrt_information), nullptr,
        submap_a.pose.data(), submap_b.pose.data());
    prior_residual_blocks.insert(residual_block_id);
  }
}

void AddLoopClosureConstraints(ceres::Problem &problem,
                               std::vector<TimestampedPointCloud> &submaps,
                               const proto::PgoConfig &config) {
  Eigen::DiagonalMatrix<double, 6> loop_sqrt_information;
  loop_sqrt_information.diagonal() << 1 / config.loop_edge_translation_error(),
      1 / config.loop_edge_translation_error(),
      1 / config.loop_edge_translation_error(),
      1 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.loop_edge_rotation_error() / 180.0 * M_PI);

  std::set<std::pair<int, int>> edges_used;
  int loop_edge_num = 40;  // todo
  do {
    int target_id = rand() % submaps.size();

    // select candidates
    std::vector<int> candidate_scan_ids;
    for (int i = 0; i < submaps.size(); ++i) {
      if (abs(submaps[target_id].timestamp - submaps[i].timestamp) > config.loop_closure_search_time_diff()) {
        if ((submaps[target_id].pose.translation() -
             submaps[i].pose.translation())
                .norm() < config.loop_closure_search_radius()) {  // todo
          candidate_scan_ids.push_back(i);
        }
      }
    }

    if (candidate_scan_ids.empty()) {
      spdlog::info("candidate scan ids empty");
      continue;
    }

    if (candidate_scan_ids.size() <= 0) {
      spdlog::error("candidate_scan_ids.size() <= 0");
      exit(1);
    }
    int source_id = candidate_scan_ids[rand() % candidate_scan_ids.size()];

    if (edges_used.find({target_id, source_id}) != edges_used.end() ||
        edges_used.find({source_id, target_id}) != edges_used.end()) {
      continue;
    }

    // match
    Sophus::SE3d T_source_to_target(submaps[target_id].pose.inverse() *
                                    submaps[source_id].pose);

    if (source_id < 0) {
      spdlog::error("CHECK_GE failed: source_id >= 0");
      exit(1);
    }
    if (source_id >= submaps.size()) {
      spdlog::error("CHECK_LT failed: source_id < submaps.size()");
      exit(1);
    }
    if (target_id < 0) {
      spdlog::error("CHECK_GE failed: target_id >= 0");
      exit(1);
    }
    if (target_id >= submaps.size()) {
      spdlog::error("CHECK_LT failed: target_id < submaps.size()");
      exit(1);
    }

    if (MatchGICP(submaps[target_id].cloud, submaps[source_id].cloud,
                  T_source_to_target, config.gicp_fitness_score_threshold(),
                  config.gicp_max_iterations(),
                  config.gicp_transform_epsilon())) {
      // add loop constraint
      problem.AddResidualBlock(PoseGraphEdgeFactor::Create(
                                   T_source_to_target, loop_sqrt_information),
                               nullptr, submaps[target_id].pose.data(),
                               submaps[source_id].pose.data());

      edges_used.insert({target_id, source_id});
      loop_edge_num--;
    }
  } while (loop_edge_num > 0);
}

void RemoveNonPriorConstraints(
    ceres::Problem &problem,
    std::set<ceres::ResidualBlockId> &prior_residual_blocks) {
  std::vector<ceres::ResidualBlockId> all_residual_blocks;
  problem.GetResidualBlocks(&all_residual_blocks);
  for (auto &residual_block_id : all_residual_blocks) {
    if (prior_residual_blocks.find(residual_block_id) ==
        prior_residual_blocks.end()) {
      problem.RemoveResidualBlock(residual_block_id);
    }
  }
}

}  // namespace

namespace {

void AddGnssConstraints(
    ceres::Problem &problem,
    std::vector<TimestampedPointCloud> &submaps,
    const LocalENUTransformer &transformer,
    const std::vector<GpsData> &gnss_data,
    std::set<ceres::ResidualBlockId> &prior_residual_blocks) {
  if (gnss_data.empty()) {
    spdlog::warn("No GNSS data provided, skipping GNSS constraints");
    return;
  }

  int gnss_constraints_added = 0;

  // For each submap, find GNSS data points that bracket its timestamp and interpolate
  for (size_t submap_idx = 0; submap_idx < submaps.size(); ++submap_idx) {
    double submap_timestamp = submaps[submap_idx].timestamp;

    // Use lower_bound to find the first GNSS point >= submap_timestamp
    auto next_it = std::lower_bound(
        gnss_data.begin(), gnss_data.end(), submap_timestamp,
        [](const GpsData &gps, double timestamp) { return gps.timestamp < timestamp; });

    int next_idx = std::distance(gnss_data.begin(), next_it);
    int prev_idx = next_idx - 1;

    // Check if we have valid bracketing points
    if (prev_idx < 0 || next_idx >= gnss_data.size()) {
      continue;  // No bracketing points available
    }

    // Interpolation case: check time gap constraint (must be ≤ 0.3s)
    double time_gap = gnss_data[next_idx].timestamp - gnss_data[prev_idx].timestamp;
    if (time_gap > 0.3) {
      continue;  // Time gap too large for reliable interpolation
    }

    // Linear interpolation between two GNSS points
    double t_prev = gnss_data[prev_idx].timestamp;
    double t_next = gnss_data[next_idx].timestamp;
    double alpha  = (submap_timestamp - t_prev) / (t_next - t_prev);

    // Interpolate latitude, longitude, altitude
    double interp_lat = gnss_data[prev_idx].latitude +
                        alpha * (gnss_data[next_idx].latitude - gnss_data[prev_idx].latitude);
    double interp_lon = gnss_data[prev_idx].longitude +
                        alpha * (gnss_data[next_idx].longitude - gnss_data[prev_idx].longitude);
    double interp_alt = gnss_data[prev_idx].altitude +
                        alpha * (gnss_data[next_idx].altitude - gnss_data[prev_idx].altitude);

    // Interpolate uncertainty (use first GNSS point's std)
    double interp_lat_std = gnss_data[prev_idx].lat_std;
    double interp_lon_std = gnss_data[prev_idx].lon_std;
    double interp_alt_std = gnss_data[prev_idx].alt_std;

    // Convert interpolated GNSS to ENU
    Eigen::Vector3d enu_gps = transformer.Convert(interp_lat, interp_lon, interp_alt);

    Eigen::Vector3d position_std(interp_lat_std, interp_lon_std, interp_alt_std);


    spdlog::info("enu {} {} {} {} {} {}", enu_gps[0], enu_gps[1], enu_gps[2], submaps[submap_idx].pose.translation().x(),
                 submaps[submap_idx].pose.translation().y(), submaps[submap_idx].pose.translation().z());
    auto position_factor = RtkPositionFactor::Create(enu_gps, position_std);
    auto residual_id     = problem.AddResidualBlock(
        position_factor, nullptr, submaps[submap_idx].pose.data());
    prior_residual_blocks.insert(residual_id);
    gnss_constraints_added++;
  }

  spdlog::info("Added {} GNSS constraints to {} submaps",
               gnss_constraints_added, submaps.size());
}

}  // namespace

void OptimizeWithGnss(std::vector<TimestampedPointCloud> &submaps,
                      const std::vector<GpsData> &gnss_data,
                      const proto::PgoConfig &config,
                      bool use_rtk,
                      bool use_btc,
                      std::string &proj4_string) {
  if (submaps.empty()) {
    spdlog::error("No submaps to optimize");
    return;
  }

  if (use_rtk) {
    spdlog::info("GNSS data found ({}), using RTK fusion optimization", gnss_data.size());
  } else {
    spdlog::info("No GNSS data or RTK disabled, using standard PGO optimization");
  }

  if (use_btc) {
    spdlog::info("BTC-based loop closure constraints enabled");
  }

  ceres::Problem problem;
  std::set<ceres::ResidualBlockId> prior_residual_blocks;

  // Setup optimization
  AddParameters(problem, submaps);
  AddAdjacentConstraints(problem, submaps, prior_residual_blocks, config);
  // AddGravityConstraits(problem, submaps, prior_residual_blocks, config);

  // Add GNSS constraints only if enabled
  if (use_rtk && !gnss_data.empty()) {
    LocalENUTransformer transformer(gnss_data[0].latitude, gnss_data[0].longitude);
    proj4_string = transformer.GetProj4String();
    AddGnssConstraints(problem, submaps, transformer, gnss_data, prior_residual_blocks);
  }

  // Add BTC constraints if enabled
  if (use_btc) {
    AddBTCConstraints(problem, submaps, config);
  }

  // Iterative optimization with loop closure and GNSS updates
  for (int iter = 0; iter < config.outer_iteration_num(); ++iter) {
    // AddLoopClosureConstraints(problem, submaps, config);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    options.max_num_iterations           = config.inner_iteration_num();
    ceres::Solver::Summary summary;

    spdlog::info("Optimization iteration {} (with GNSS fusion)...", iter + 1);
    ceres::Solve(options, &problem, &summary);
    spdlog::info("{}", summary.FullReport());

    // Clean up non-prior constraints for the next iteration
    // RemoveNonPriorConstraints(problem, prior_residual_blocks);
  }

  spdlog::info("GNSS-fused optimization completed successfully");
}

void AddBTCConstraints(ceres::Problem &problem,
                       std::vector<TimestampedPointCloud> &submaps,
                       const proto::PgoConfig &config) {
  if (submaps.size() < 2) {
    spdlog::warn("Not enough submaps for BTC constraint detection: {}", submaps.size());
    return;
  }

  spdlog::info("Starting BTC-based loop closure detection on {} submaps", submaps.size());

  // ---------- 1. Configure STDescManager ----------
  // Use indoor-friendly defaults. skip_near_num controls how many
  // temporally-adjacent frames are excluded during retrieval; it is set
  // proportionally to loop_closure_search_time_diff / submap spacing.
  ConfigSetting btc_cfg;
  btc_cfg.voxel_size_                 = 1.0f;
  btc_cfg.voxel_init_num_             = 10;
  btc_cfg.plane_merge_normal_thre_    = 0.1f;
  btc_cfg.plane_merge_dis_thre_       = 0.3f;
  btc_cfg.plane_detection_thre_       = 0.01f;
  btc_cfg.proj_plane_num_             = 2;
  btc_cfg.proj_image_resolution_      = 0.5f;
  btc_cfg.proj_image_high_inc_        = 0.1f;
  btc_cfg.proj_dis_min_               = 0.0f;
  btc_cfg.proj_dis_max_               = 5.0f;
  btc_cfg.summary_min_thre_           = 10.0f;
  btc_cfg.line_filter_enable_         = 1;
  btc_cfg.useful_corner_num_          = 100;
  btc_cfg.descriptor_near_num_        = 15.0f;
  btc_cfg.descriptor_min_len_         = 2.0f;
  btc_cfg.descriptor_max_len_         = 50.0f;
  btc_cfg.non_max_suppression_radius_ = 2.0f;
  btc_cfg.std_side_resolution_        = 0.2f;
  // skip_near_num: skip this many *frames* during candidate retrieval.
  // Each submap is ~1 s; loop_closure_search_time_diff is given in seconds.
  btc_cfg.skip_near_num_              = static_cast<int>(config.loop_closure_search_time_diff());
  btc_cfg.candidate_num_              = 20;
  btc_cfg.rough_dis_threshold_        = 0.01f;
  btc_cfg.similarity_threshold_       = 0.7f;
  btc_cfg.icp_threshold_              = 0.15f;
  btc_cfg.normal_threshold_           = 0.2f;
  btc_cfg.dis_threshold_              = 0.5f;

  STDescManager desc_manager(btc_cfg);

  // Information matrix for BTC loop-closure constraints
  Eigen::DiagonalMatrix<double, 6> btc_sqrt_information;
  btc_sqrt_information.diagonal() <<
      1.0 / config.loop_edge_translation_error(),
      1.0 / config.loop_edge_translation_error(),
      1.0 / config.loop_edge_translation_error(),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI);

  int btc_constraints_added = 0;

  // ---------- 2. Online build+search loop ----------
  // Process submaps in chronological order.
  // For each frame i:
  //   a) Generate STDescs from its point cloud.
  //   b) SearchLoop against the database built so far (frames 0..i-1).
  //      The library skips frames whose frame_number is within skip_near_num of
  //      the current frame, so only temporally distant matches are returned.
  //   c) If a candidate is found and passes spatial filter, add constraint.
  //   d) AddSTDescs to register this frame in the database.
  for (int i = 0; i < static_cast<int>(submaps.size()); ++i) {
    if (!submaps[i].cloud || submaps[i].cloud->empty()) {
      // Still need to advance the database frame counter
      std::vector<STD> empty_stds;
      desc_manager.AddSTDescs(empty_stds);
      continue;
    }

    // a) Generate descriptors for this submap
    std::vector<STD> stds;
    desc_manager.GenerateSTDescs(submaps[i].cloud, stds, i);

    // b) Search for a loop closure candidate among already-registered frames.
    //    SearchLoop internally respects skip_near_num so only frames that are
    //    at least skip_near_num frames away are considered.
    if (i > btc_cfg.skip_near_num_) {
      std::pair<int, double>                              loop_result;
      std::pair<Eigen::Vector3d, Eigen::Matrix3d>         loop_transform;
      std::vector<std::pair<STD, STD>>                    loop_std_pair;
      // plane cloud of the current scan (needed by SearchLoop's icp verify)
      pcl::PointCloud<pcl::PointXYZINormal>::Ptr pl_cur(new pcl::PointCloud<pcl::PointXYZINormal>());

      desc_manager.SearchLoop(stds, loop_result, loop_transform, loop_std_pair, pl_cur);

      int matched_id = loop_result.first;  // -1 means no match
      if (matched_id >= 0 && matched_id < i) {
        // c) Verify with additional spatial distance filter
        double spatial_dist =
            (submaps[i].pose.translation() - submaps[matched_id].pose.translation()).norm();
        if (spatial_dist <= config.loop_closure_search_radius()) {
          // Build the relative pose SE3 from the BTC transform
          // loop_transform: (translation, rotation_matrix) that maps
          // the matched frame to the current frame.
          const Eigen::Vector3d &t_btc = loop_transform.first;
          const Eigen::Matrix3d &R_btc = loop_transform.second;
          Sophus::SE3d T_matched_to_cur(R_btc, t_btc);

          problem.AddResidualBlock(
              BTCFactor::Create(T_matched_to_cur, btc_sqrt_information),
              nullptr,
              submaps[i].pose.data(),
              submaps[matched_id].pose.data());

          btc_constraints_added++;
          spdlog::info("BTC constraint: submap {} <-> {} (score={:.3f}, dist={:.2f}m)",
                       i, matched_id, loop_result.second, spatial_dist);
        } else {
          spdlog::debug("BTC match {}<->{} rejected: spatial dist {:.2f} > {:.2f}",
                        i, matched_id, spatial_dist, config.loop_closure_search_radius());
        }
      }
    }

    // d) Register this frame's descriptors in the database
    desc_manager.AddSTDescs(stds);
  }

  spdlog::info("BTC constraint detection finished: {} constraints added", btc_constraints_added);
}
