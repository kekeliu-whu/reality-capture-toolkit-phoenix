#define SOPHUS_DISABLE_ENSURES

#include <algorithm>
#include <cctype>
#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <set>
#include <thread>
#include <vector>

#include <ceres/ceres.h>
#include <pcl/common/transforms.h>
#include <pcl/io/pcd_io.h>
#include <pcl/search/kdtree.h>
#include <spdlog/spdlog.h>

// BTC library files are located directly in pgo/src/.
// Use btc_compat/ stubs to satisfy ros/ros.h and visualization_msgs headers.
#include "BTC.h"

#include "factor/gravity_factor.h"
#include "factor/local_parameterization_se3.h"
#include "factor/pose_graph_edge_factor.h"
#include "factor/rtk_factor.h"
#include "io/local_enu_transformer.h"
#include "io/read_write.h"
#include "optimizer.h"

namespace {

using PointT = pcl::PointXYZI;

// ---------------------------------------------------------------------------
// Ceres point-to-plane residual
// ---------------------------------------------------------------------------
struct P2PlResidual {
  P2PlResidual(const Eigen::Vector3d &src, const Eigen::Vector3d &ctr,
               const Eigen::Vector3d &nrm)
      : src_(src), ctr_(ctr), nrm_(nrm) {}

  template <typename T>
  bool operator()(const T *const pose, T *r) const {
    Eigen::Map<const Sophus::SE3<T>> tf(pose);
    r[0] = ((tf * src_.cast<T>()) - ctr_.cast<T>()).dot(nrm_.cast<T>());
    return true;
  }

  static ceres::CostFunction *Make(const Eigen::Vector3d &s,
                                   const Eigen::Vector3d &c,
                                   const Eigen::Vector3d &n) {
    return new ceres::AutoDiffCostFunction<P2PlResidual, 1, 7>(
        new P2PlResidual(s, c, n));
  }

  Eigen::Vector3d src_, ctr_, nrm_;
};

// ---------------------------------------------------------------------------
// PCA plane fitting: (center, normal) from 8 nearest neighbors
// ---------------------------------------------------------------------------
struct PlaneFit {
  Eigen::Vector3d center{Eigen::Vector3d::Zero()};
  Eigen::Vector3d normal{Eigen::Vector3d::UnitZ()};
  bool valid{false};

  static PlaneFit FromPoints(const std::vector<Eigen::Vector3d> &pts) {
    PlaneFit p;
    if (pts.size() < 3) return p;
    for (const auto &v : pts) p.center += v;
    p.center /= pts.size();

    Eigen::Matrix3d cov = Eigen::Matrix3d::Zero();
    for (const auto &v : pts) {
      Eigen::Vector3d d = v - p.center;
      cov += d * d.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(cov);
    if (eig.info() != Eigen::Success) return p;

    const auto &ev = eig.eigenvalues();
    int k          = (ev(1) < ev(0)) ? 1 : 0;
    k              = (ev(2) < ev(k)) ? 2 : k;
    p.normal       = eig.eigenvectors().col(k).normalized();
    p.valid        = true;
    return p;
  }
};

// ---------------------------------------------------------------------------
// Correspondence: source point → target plane (center + normal)
// ---------------------------------------------------------------------------
struct Corr {
  Eigen::Vector3d src_pt, center, normal;
};

void BuildCorrespondences(
    const pcl::PointCloud<PointT> &source,
    const pcl::PointCloud<PointT> &target,
    const Sophus::SE3d &T,
    const pcl::search::KdTree<PointT> &kdtree,
    double max_p2p_dist,
    std::vector<Corr> &corrs) {
  corrs.clear();
  constexpr int K = 8;
  std::vector<int> knn_idx(K);
  std::vector<float> knn_d2(K);
  int n_p2p = 0, n_invalid = 0;

  for (size_t i = 0; i < source.size(); ++i) {
    const auto &pt = source[i];
    if (!std::isfinite(pt.x) || !std::isfinite(pt.y) || !std::isfinite(pt.z)) {
      n_invalid++;
      continue;
    }

    PointT pt_t;
    Eigen::Vector3d p_src(pt.x, pt.y, pt.z);
    Eigen::Vector3d p_tgt = T * p_src;
    pt_t.x                = static_cast<float>(p_tgt.x());
    pt_t.y                = static_cast<float>(p_tgt.y());
    pt_t.z                = static_cast<float>(p_tgt.z());

    kdtree.nearestKSearch(pt_t, K, knn_idx, knn_d2);

    std::vector<Eigen::Vector3d> nbrs(K);
    for (int j = 0; j < K; ++j) {
      const auto &np = target[knn_idx[j]];
      nbrs[j]        = {np.x, np.y, np.z};
    }
    PlaneFit plane = PlaneFit::FromPoints(nbrs);
    if (!plane.valid) continue;

    double d = std::abs((p_tgt - plane.center).dot(plane.normal));
    if (d > max_p2p_dist) {
      n_p2p++;
      continue;
    }

    corrs.push_back({p_src, plane.center, plane.normal});
  }

  spdlog::info("  p2pl corrs: {} accepted, {} rejected / {}",
               corrs.size(), n_p2p, source.size());
}

Sophus::SE3d SolveP2Pl(const std::vector<Corr> &corrs,
                       const Sophus::SE3d &T_init) {
  double pose[7];
  Eigen::Map<Eigen::Quaterniond> q(pose);
  Eigen::Map<Eigen::Vector3d> t(pose + 4);
  q = T_init.unit_quaternion();
  t = T_init.translation();

  ceres::Problem problem;
  problem.AddParameterBlock(pose, 7, LocalParameterizationSE3::Create());

  ceres::LossFunction *loss = new ceres::HuberLoss(0.1);
  for (const auto &c : corrs) {
    problem.AddResidualBlock(P2PlResidual::Make(c.src_pt, c.center, c.normal),
                             loss, pose);
  }

  ceres::Solver::Options opts;
  opts.linear_solver_type           = ceres::DENSE_QR;
  opts.max_num_iterations           = 10;
  opts.num_threads                  = std::thread::hardware_concurrency();
  opts.function_tolerance           = 1e-8;
  opts.minimizer_progress_to_stdout = false;

  ceres::Solver::Summary sum;
  ceres::Solve(opts, &problem, &sum);

  spdlog::info("  p2pl ceres: {} iters, cost {:.2f} -> {:.2f}",
               sum.iterations.size(), sum.initial_cost, sum.final_cost);

  return Sophus::SE3d(q, t);
}

// ---------------------------------------------------------------------------

std::string GetEnvString(const char *name) {
  const char *value = nullptr;
#ifdef _MSC_VER
  char *buffer       = nullptr;
  size_t buffer_size = 0;
  if (_dupenv_s(&buffer, &buffer_size, name) != 0 || buffer == nullptr) {
    return "";
  }
  value = buffer;
#else
  value = std::getenv(name);
  if (value == nullptr) {
    return "";
  }
#endif

  std::string result(value);
#ifdef _MSC_VER
  free(buffer);
#endif
  return result;
}

bool IsEnabledFromEnv(const char *name) {
  std::string env_value = GetEnvString(name);
  if (env_value.empty()) {
    return false;
  }

  std::string normalized(std::move(env_value));
  std::transform(normalized.begin(), normalized.end(), normalized.begin(),
                 [](unsigned char c) {
                   return static_cast<char>(std::tolower(c));
                 });
  return normalized == "1" || normalized == "true" ||
         normalized == "yes" || normalized == "on";
}

double EstimateMedianTimestampGap(
    const std::vector<TimestampedPointCloud> &submaps) {
  if (submaps.size() < 2) {
    return 0.0;
  }

  std::vector<double> gaps;
  gaps.reserve(submaps.size() - 1);
  for (size_t i = 1; i < submaps.size(); ++i) {
    const double gap = submaps[i].timestamp - submaps[i - 1].timestamp;
    if (gap > 0.0) {
      gaps.push_back(gap);
    }
  }

  if (gaps.empty()) {
    return 0.0;
  }

  const size_t mid = gaps.size() / 2;
  std::nth_element(gaps.begin(), gaps.begin() + mid, gaps.end());
  if ((gaps.size() % 2) == 1) {
    return gaps[mid];
  }

  const double upper = gaps[mid];
  std::nth_element(gaps.begin(), gaps.begin() + mid - 1, gaps.end());
  return 0.5 * (gaps[mid - 1] + upper);
}

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

double PoseDistance3D(const TimestampedPose &a, const TimestampedPose &b) {
  return (a.pose->translation() - b.pose->translation()).norm();
}

double PoseDistance2D(const TimestampedPose &a, const TimestampedPose &b) {
  const Eigen::Vector3d delta = a.pose->translation() - b.pose->translation();
  return delta.head<2>().norm();
}

void FillTrajectoryMetrics(
    const std::vector<TimestampedPose> &timestamped_scan_poses,
    proto::PgoMetrics *metrics) {
  if (metrics == nullptr || timestamped_scan_poses.empty()) {
    return;
  }

  metrics->set_pose_count(static_cast<int>(timestamped_scan_poses.size()));
  metrics->set_trajectory_start_timestamp(
      timestamped_scan_poses.front().timestamp);
  metrics->set_trajectory_end_timestamp(
      timestamped_scan_poses.back().timestamp);
  metrics->set_trajectory_duration_secs(
      std::max(0.0, timestamped_scan_poses.back().timestamp -
                        timestamped_scan_poses.front().timestamp));

  double path_length_3d = 0.0;
  double path_length_2d = 0.0;
  for (size_t i = 1; i < timestamped_scan_poses.size(); ++i) {
    path_length_3d += PoseDistance3D(timestamped_scan_poses[i],
                                     timestamped_scan_poses[i - 1]);
    path_length_2d += PoseDistance2D(timestamped_scan_poses[i],
                                     timestamped_scan_poses[i - 1]);
  }
  metrics->set_path_length_3d_m(path_length_3d);
  metrics->set_path_length_2d_m(path_length_2d);
  metrics->set_start_end_distance_3d_m(
      PoseDistance3D(timestamped_scan_poses.back(),
                     timestamped_scan_poses.front()));
  metrics->set_start_end_distance_2d_m(
      PoseDistance2D(timestamped_scan_poses.back(),
                     timestamped_scan_poses.front()));
}

bool MatchGICP(pcl::PointCloud<pcl::PointXYZI>::Ptr &target,
               pcl::PointCloud<pcl::PointXYZI>::Ptr &source,
               Sophus::SE3d &T_source_to_target,
               double gicp_fitness_score_threshold,
               int /*gicp_max_iterations*/,
               double /*gicp_transform_epsilon*/,
               pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_source_out = nullptr) {
  if (!source || !target) {
    spdlog::error("MatchGICP: null cloud");
    return false;
  }

  pcl::search::KdTree<PointT>::Ptr kdtree(new pcl::search::KdTree<PointT>());
  kdtree->setInputCloud(target);

  constexpr double kMaxP2PlDist = 0.4;

  std::vector<Corr> corrs;
  for (int iter = 0; iter < 3; ++iter) {
    BuildCorrespondences(*source, *target, T_source_to_target,
                         *kdtree, kMaxP2PlDist, corrs);
    if (corrs.size() < 50) {
      spdlog::warn("MatchGICP: too few correspondences ({})", corrs.size());
      return false;
    }

    Sophus::SE3d T_new = SolveP2Pl(corrs, T_source_to_target);
    T_source_to_target = T_new;
  }

  if (aligned_source_out) {
    pcl::transformPointCloud(*source, *aligned_source_out,
                             T_source_to_target.matrix().cast<float>());
  }

  double score = CalcFitnessScore(target, source, T_source_to_target, 2.0);
  spdlog::info("p2pl match score: {:.4f}", score);
  return score < gicp_fitness_score_threshold;
}

struct BTCConstraintStats {
  int constraints_added          = 0;
  int frames_without_descriptors = 0;
  int frames_without_candidate   = 0;
  int matches_rejected_by_time   = 0;
  int matches_rejected_by_gicp   = 0;
  int matches_rejected_by_sanity = 0;
  int accepted_outside_radius    = 0;
};

BTCConstraintStats AddBTCConstraintsInternal(
    ceres::Problem &problem,
    std::vector<TimestampedPointCloud> &submaps,
    const proto::PgoConfig &config,
    proto::PgoMetrics *metrics);

ConfigSetting MakeBTCConfig(const proto::PgoConfig &config,
                            double median_timestamp_gap) {
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
  btc_cfg.touch_filter_enable_        = 0;
  btc_cfg.descriptor_near_num_        = 15.0f;
  btc_cfg.descriptor_min_len_         = 2.0f;
  btc_cfg.descriptor_max_len_         = 50.0f;
  btc_cfg.non_max_suppression_radius_ = 2.0f;
  btc_cfg.std_side_resolution_        = 0.2f;
  // Road and corridor data often produces many similar, low-texture planes.
  // The old retrieval gate (20 candidates, 0.7 similarity, 0.01 rough
  // distance) discarded nearly every real revisit before the geometric
  // verifier could inspect it.  Keep retrieval permissive and let the
  // point-to-plane matcher below reject false closures.
  btc_cfg.skip_near_num_              = 15;
  btc_cfg.candidate_num_              = 100;
  btc_cfg.rough_dis_threshold_        = 0.03f;
  btc_cfg.similarity_threshold_       = 0.5f;
  btc_cfg.icp_threshold_              = 0.15f;
  btc_cfg.normal_threshold_           = 0.2f;
  btc_cfg.dis_threshold_              = 0.5f;

  if (median_timestamp_gap > 0.0) {
    btc_cfg.skip_near_num_ = std::max(
        10, static_cast<int>(std::llround(
                config.loop_closure_search_time_diff() / median_timestamp_gap)));
  }

  return btc_cfg;
}

void LogBTCConfig(const ConfigSetting &btc_cfg,
                  const proto::PgoConfig &config,
                  double median_timestamp_gap,
                  double trajectory_duration,
                  bool visualize_stdescs) {
  if (median_timestamp_gap <= 0.0) {
    spdlog::warn(
        "BTC could not infer a positive timestamp gap from submaps; using default skip_near_num={} frames",
        btc_cfg.skip_near_num_);
  }

  spdlog::info(
      "BTC config: median_dt={:.3f}s, traj_duration={:.1f}s, skip_near_num={} frames, candidate_num={}, similarity_threshold={:.2f}, icp_threshold={:.2f}",
      median_timestamp_gap, trajectory_duration, btc_cfg.skip_near_num_,
      btc_cfg.candidate_num_, btc_cfg.similarity_threshold_,
      btc_cfg.icp_threshold_);

  if (trajectory_duration <= config.loop_closure_search_time_diff()) {
    spdlog::warn(
        "Trajectory duration {:.1f}s is not longer than loop_closure_search_time_diff {:.1f}s. BTC retrieval may never accept a time-separated loop for this dataset.",
        trajectory_duration, config.loop_closure_search_time_diff());
  }

  if (visualize_stdescs) {
    spdlog::info(
        "PGO_BTC_VISUALIZE is enabled. BTC descriptor visualization will block on each submap until you press N.");
  }
}

Eigen::DiagonalMatrix<double, 6> MakeBTCSqrtInformation(
    const proto::PgoConfig &config) {
  Eigen::DiagonalMatrix<double, 6> btc_sqrt_information;
  btc_sqrt_information.diagonal()
      << 1.0 / config.loop_edge_translation_error(),
      1.0 / config.loop_edge_translation_error(),
      1.0 / config.loop_edge_translation_error(),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI),
      1.0 / (config.loop_edge_rotation_error() / 180.0 * M_PI);
  return btc_sqrt_information;
}

pcl::PointCloud<pcl::PointXYZINormal>::Ptr GetCurrentPlaneCloud(
    STDescManager &desc_manager) {
  if (!desc_manager.plane_cloud_vec_.empty()) {
    return desc_manager.plane_cloud_vec_.back();
  }
  return pcl::PointCloud<pcl::PointXYZINormal>::Ptr(
      new pcl::PointCloud<pcl::PointXYZINormal>());
}

void TryAddBTCConstraintForFrame(
    int frame_id,
    std::vector<STD> &stds,
    ceres::Problem &problem,
    std::vector<TimestampedPointCloud> &submaps,
    STDescManager &desc_manager,
    const ConfigSetting &btc_cfg,
    const proto::PgoConfig &config,
    const Eigen::DiagonalMatrix<double, 6> &btc_sqrt_information,
    BTCConstraintStats &stats,
    proto::PgoMetrics *metrics) {
  if (frame_id <= btc_cfg.skip_near_num_) {
    return;
  }

  std::pair<int, double> loop_result;
  std::pair<Eigen::Vector3d, Eigen::Matrix3d> loop_transform;
  std::vector<std::pair<STD, STD>> loop_std_pair;
  auto pl_cur = GetCurrentPlaneCloud(desc_manager);
  desc_manager.SearchLoop(stds, loop_result, loop_transform, loop_std_pair,
                          pl_cur);

  const int matched_id = loop_result.first;
  if (matched_id < 0) {
    ++stats.frames_without_candidate;
    return;
  }
  if (matched_id >= frame_id) {
    return;
  }

  const double time_diff =
      std::abs(submaps[frame_id].timestamp - submaps[matched_id].timestamp);
  if (time_diff <= config.loop_closure_search_time_diff()) {
    ++stats.matches_rejected_by_time;
    spdlog::info(
        "BTC match {}<->{} rejected: time diff {:.2f}s <= {:.2f}s",
        frame_id, matched_id, time_diff,
        config.loop_closure_search_time_diff());
    return;
  }

  const double spatial_dist =
      (submaps[frame_id].pose->translation() -
       submaps[matched_id].pose->translation())
          .norm();
  if (spatial_dist > config.loop_closure_search_radius()) {
    ++stats.accepted_outside_radius;
    spdlog::warn(
        "BTC match {}<->{} verified with score {:.3f} but prior pose distance is {:.2f}m > {:.2f}m. Keeping it because prior drift can hide real loops.",
        frame_id, matched_id, loop_result.second, spatial_dist,
        config.loop_closure_search_radius());
  }

  const Eigen::Vector3d &t_btc = loop_transform.first;
  const Eigen::Matrix3d &R_btc = loop_transform.second;
  Sophus::SE3d T_current_to_matched(R_btc, t_btc);
  // Save BTC rough transform before GICP overwrites it
  const Sophus::SE3d T_btc_init = T_current_to_matched;

  pcl::PointCloud<pcl::PointXYZI>::Ptr aligned_source(new pcl::PointCloud<pcl::PointXYZI>);
  const bool gicp_ok = MatchGICP(
      submaps[matched_id].cloud,
      submaps[frame_id].cloud,
      T_current_to_matched,
      config.gicp_fitness_score_threshold(),
      config.gicp_max_iterations(),
      config.gicp_transform_epsilon(),
      aligned_source);

  if (gicp_ok) {
    // Compute: original trajectory relative pose vs GICP refined pose
    const Sophus::SE3d T_original = submaps[matched_id].pose->inverse() * (*submaps[frame_id].pose);
    // T_diff = T_original * T_gicp^{-1}: error of original pose relative to GICP
    const Sophus::SE3d T_diff = T_original * T_current_to_matched.inverse();
    const Eigen::Vector3d dt  = T_diff.translation();
    const double dr_deg       = Eigen::AngleAxisd(T_diff.rotationMatrix()).angle() * 180.0 / M_PI;
    const double gicp_fitness =
        CalcFitnessScore(submaps[matched_id].cloud,
                         submaps[frame_id].cloud,
                         T_current_to_matched,
                         2.0);

    // A genuine late loop can carry several metres of accumulated drift, so
    // do not reject it merely for a large translation correction.  A nearly
    // flipped heading, however, is an unambiguous failure mode on repetitive
    // road planes: it can pass the local point-to-plane fitness check and then
    // fold the whole pose graph.  Keep a generous bound that still permits
    // normal loop corrections while blocking those catastrophic aliases.
    constexpr double kMaxLoopCorrectionTranslationM = 10.0;
    constexpr double kMaxLoopCorrectionRotationDeg  = 45.0;
    if (dt.norm() > kMaxLoopCorrectionTranslationM ||
        dr_deg > kMaxLoopCorrectionRotationDeg) {
      ++stats.matches_rejected_by_sanity;
      spdlog::warn(
          "BTC match {}<->{} rejected by correction sanity gate: translation {:.3f}m (max {:.1f}m), rotation {:.3f}deg (max {:.1f}deg)",
          frame_id, matched_id, dt.norm(), kMaxLoopCorrectionTranslationM,
          dr_deg, kMaxLoopCorrectionRotationDeg);
      return;
    }

    spdlog::info(
        "Loop edge submap {:4d} -> {:<4d} | "
        "t_orig=({:+7.3f},{:+7.3f},{:+7.3f}) L={:6.2f}m | "
        "t_gicp=({:+7.3f},{:+7.3f},{:+7.3f}) L={:6.2f}m | "
        "dt_err=({:+6.1f},{:+6.1f},{:+6.1f})mm |d|={:6.1f}mm dr_err={:6.3f}deg | "
        "score={:.3f} fitness={:.4f}",
        frame_id, matched_id,
        T_original.translation().x(), T_original.translation().y(), T_original.translation().z(),
        T_original.translation().norm(),
        T_current_to_matched.translation().x(), T_current_to_matched.translation().y(), T_current_to_matched.translation().z(),
        T_current_to_matched.translation().norm(),
        dt.x() * 1000.0, dt.y() * 1000.0, dt.z() * 1000.0, dt.norm() * 1000.0, dr_deg,
        loop_result.second,
        gicp_fitness);

    if (metrics != nullptr) {
      auto *loop_metric = metrics->add_loop_constraints();
      loop_metric->set_from_submap_index(frame_id);
      loop_metric->set_to_submap_index(matched_id);
      loop_metric->set_score(loop_result.second);
      loop_metric->set_time_diff_secs(time_diff);
      loop_metric->set_prior_distance_m(spatial_dist);
      loop_metric->set_translation_error_m(dt.norm());
      loop_metric->set_rotation_error_deg(dr_deg);
      loop_metric->set_gicp_fitness_score(gicp_fitness);
    }

    // Save debug clouds for accepted loop closures only
    const std::string debug_dir = GetEnvString("PGO_GICP_DEBUG_DIR");
    if (!debug_dir.empty()) {
      std::string pair_dir = fmt::format("{}/submap_{:04d}_to_{:04d}", debug_dir, frame_id, matched_id);
      std::filesystem::create_directories(pair_dir);
      // 1. s1 (target/matched submap) local point cloud
      pcl::io::savePCDFileBinary(pair_dir + "/cloud1_target.pcd", *submaps[matched_id].cloud);
      // 2. s2 transformed to s1 by original trajectory pose
      pcl::PointCloud<pcl::PointXYZI> source_original;
      pcl::transformPointCloud(*submaps[frame_id].cloud, source_original, T_original.matrix().cast<float>());
      pcl::io::savePCDFileBinary(pair_dir + "/cloud2_source_original_pose.pcd", source_original);
      // 3. s2 transformed to s1 by BTC rough pose
      pcl::PointCloud<pcl::PointXYZI> source_btc;
      pcl::transformPointCloud(*submaps[frame_id].cloud, source_btc, T_btc_init.matrix().cast<float>());
      pcl::io::savePCDFileBinary(pair_dir + "/cloud3_source_btc.pcd", source_btc);
      // 4. s2 transformed to s1 by GICP refined pose
      pcl::io::savePCDFileBinary(pair_dir + "/cloud4_source_gicp.pcd", *aligned_source);
    }
  }
  if (!gicp_ok) {
    ++stats.matches_rejected_by_gicp;
    spdlog::info("BTC match {}<->{} rejected by GICP refinement",
                 frame_id, matched_id);
    return;
  }

  problem.AddResidualBlock(
      PoseGraphEdgeFactor::Create(T_current_to_matched.inverse(),
                                  btc_sqrt_information.toDenseMatrix()),
      new ceres::HuberLoss(1.0),
      submaps[frame_id].pose->data(),
      submaps[matched_id].pose->data());

  ++stats.constraints_added;
  spdlog::info(
      "BTC constraint: submap {} <-> {} (score={:.3f}, time={:.2f}s, prior_dist={:.2f}m, stds={})",
      frame_id, matched_id, loop_result.second, time_diff, spatial_dist,
      stds.size());
}

void AddParameters(ceres::Problem &problem,
                   std::vector<TimestampedPose> &timestamped_scan_poses) {
  ceres::Manifold *local_parameterization =
      LocalParameterizationSE3::Create();

  for (auto &scan_pose : timestamped_scan_poses) {
    problem.AddParameterBlock(scan_pose.pose->data(), 7, local_parameterization);
  }
}

int AddGravityConstraints(ceres::Problem &problem,
                          std::vector<TimestampedPose> &timestamped_scan_poses,
                          std::vector<TimestampedPointCloud> &submaps,
                          std::set<ceres::ResidualBlockId> &prior_residual_blocks,
                          const proto::PgoConfig &config) {
  if (!config.use_gravity_alignment()) {
    spdlog::info("Gravity alignment is disabled by config");
    return 0;
  }

  const Eigen::Vector3d kGravityRef(0.0, 0.0, -1.0);
  int constraints_added = 0;

  for (size_t submap_index = 0; submap_index < submaps.size(); ++submap_index) {
    auto &submap = submaps[submap_index];
    if (!submap.pose) {
      spdlog::warn("Skipping gravity constraint for submap {}: missing pose",
                   submap_index);
      continue;
    }

    if (submap.anchor_pose_index >= timestamped_scan_poses.size()) {
      spdlog::warn("Skipping gravity constraint for submap {}: invalid anchor pose index {}",
                   submap_index,
                   submap.anchor_pose_index);
      continue;
    }

    auto &anchor_pose = timestamped_scan_poses[submap.anchor_pose_index];
    const double gravity_norm = anchor_pose.gravity.norm();
    if (!std::isfinite(gravity_norm) || gravity_norm <= 1e-9) {
      spdlog::warn("Skipping gravity constraint for submap {} at timestamp {:.6f}: invalid gravity ({}, {}, {})",
                   submap_index,
                   submap.timestamp,
                   anchor_pose.gravity.x(),
                   anchor_pose.gravity.y(),
                   anchor_pose.gravity.z());
      continue;
    }

    Eigen::Vector3d g_measured_body =
        submap.pose->so3().inverse() * anchor_pose.gravity / gravity_norm;

    auto residual_block_id = problem.AddResidualBlock(
        GravityCostFunctor::Create(g_measured_body, kGravityRef,
                                   config.gravity_align_rotation_error()),
        nullptr, submap.pose->data());
    prior_residual_blocks.insert(residual_block_id);
    ++constraints_added;
  }
  spdlog::info("Added gravity constraints for {} submaps",
               constraints_added);
  return constraints_added;
}

int AddAdjacentConstraints(
    ceres::Problem &problem,
    std::vector<TimestampedPose> &timestamped_scan_poses,
    std::set<ceres::ResidualBlockId> &prior_residual_blocks,
    const proto::PgoConfig &config) {
  Eigen::DiagonalMatrix<double, 6> odom_sqrt_information;
  odom_sqrt_information.diagonal() << 1 / config.odom_edge_translation_error(),
      1 / config.odom_edge_translation_error(),
      1 / config.odom_edge_translation_error(),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI),
      1 / (config.odom_edge_rotation_error() / 180.0 * M_PI);

  int constraints_added = 0;
  for (size_t i = 0; i + 1 < timestamped_scan_poses.size(); ++i) {
    auto &scan_pose_a = timestamped_scan_poses[i];
    auto &scan_pose_b = timestamped_scan_poses[i + 1];

    Sophus::SE3d T_b2a     = scan_pose_a.pose->inverse() * (*scan_pose_b.pose);
    auto residual_block_id = problem.AddResidualBlock(
        PoseGraphEdgeFactor::Create(T_b2a, odom_sqrt_information), nullptr,
        scan_pose_a.pose->data(), scan_pose_b.pose->data());
    prior_residual_blocks.insert(residual_block_id);
    ++constraints_added;
  }
  return constraints_added;
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

int AddGnssConstraints(
    ceres::Problem &problem,
    std::vector<TimestampedPose> &timestamped_scan_poses,
    const LocalENUTransformer &transformer,
    const std::vector<GpsData> &gnss_data,
    std::set<ceres::ResidualBlockId> &prior_residual_blocks) {
  if (gnss_data.empty()) {
    spdlog::warn("No GNSS data provided, skipping GNSS constraints");
    return 0;
  }

  int gnss_constraints_added = 0;

  // For each scan pose, find GNSS data points that bracket its timestamp and interpolate
  for (size_t scan_idx = 0; scan_idx < timestamped_scan_poses.size(); ++scan_idx) {
    double scan_timestamp = timestamped_scan_poses[scan_idx].timestamp;

    // Use lower_bound to find the first GNSS point >= scan timestamp
    auto next_it = std::lower_bound(
        gnss_data.begin(), gnss_data.end(), scan_timestamp,
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
    double alpha  = (scan_timestamp - t_prev) / (t_next - t_prev);

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

    spdlog::info("enu {} {} {} {} {} {}", enu_gps[0], enu_gps[1], enu_gps[2], timestamped_scan_poses[scan_idx].pose->translation().x(),
                 timestamped_scan_poses[scan_idx].pose->translation().y(), timestamped_scan_poses[scan_idx].pose->translation().z());
    auto position_factor = RtkPositionFactor::Create(enu_gps, position_std);
    auto residual_id     = problem.AddResidualBlock(
        position_factor, nullptr, timestamped_scan_poses[scan_idx].pose->data());
    prior_residual_blocks.insert(residual_id);
    gnss_constraints_added++;
  }

  spdlog::info("Added {} GNSS constraints to {} scan poses",
               gnss_constraints_added, timestamped_scan_poses.size());
  return gnss_constraints_added;
}

}  // namespace

void Optimize(std::vector<TimestampedPose> &timestamped_scan_poses,
              std::vector<TimestampedPointCloud> &submaps,
              const proto::PgoConfig &config) {
  const std::vector<GpsData> empty_gnss_data;
  std::string proj4_string;
  OptimizeWithGnss(timestamped_scan_poses,
                   submaps,
                   empty_gnss_data,
                   config,
                   false,
                   proj4_string);
}

void OptimizeWithGnss(std::vector<TimestampedPose> &timestamped_scan_poses,
                      std::vector<TimestampedPointCloud> &submaps,
                      const std::vector<GpsData> &gnss_data,
                      const proto::PgoConfig &config,
                      bool use_rtk,
                      std::string &proj4_string,
                      proto::PgoMetrics *metrics) {
  if (timestamped_scan_poses.empty() || submaps.empty()) {
    spdlog::error("No scan poses or submaps to optimize");
    return;
  }

  if (use_rtk) {
    spdlog::info("GNSS data found ({}), using RTK fusion optimization", gnss_data.size());
  } else {
    spdlog::info("No GNSS data or RTK disabled, using standard PGO optimization");
  }

  ceres::Problem problem;
  std::set<ceres::ResidualBlockId> prior_residual_blocks;

  if (metrics != nullptr) {
    metrics->Clear();
    metrics->set_schema_version("pgo_metrics.v1");
    metrics->set_pose_count(static_cast<int>(timestamped_scan_poses.size()));
    metrics->set_submap_count(static_cast<int>(submaps.size()));
    metrics->set_use_rtk(use_rtk);
    metrics->set_gnss_measurement_count(static_cast<int>(gnss_data.size()));
  }

  // Setup optimization
  AddParameters(problem, timestamped_scan_poses);
  const int adjacent_constraint_count =
      AddAdjacentConstraints(problem, timestamped_scan_poses, prior_residual_blocks, config);
  const int gravity_constraint_count =
      AddGravityConstraints(problem, timestamped_scan_poses, submaps, prior_residual_blocks, config);
  if (metrics != nullptr) {
    metrics->set_adjacent_constraint_count(adjacent_constraint_count);
    metrics->set_gravity_constraint_count(gravity_constraint_count);
  }

  // Add GNSS constraints only if enabled
  int gnss_constraint_count = 0;
  if (use_rtk && !gnss_data.empty()) {
    LocalENUTransformer transformer(gnss_data[0].latitude, gnss_data[0].longitude);
    proj4_string = transformer.GetProj4String();
    gnss_constraint_count =
        AddGnssConstraints(problem, timestamped_scan_poses, transformer, gnss_data, prior_residual_blocks);
  }
  if (metrics != nullptr) {
    metrics->set_gnss_constraint_count(gnss_constraint_count);
  }

  // Add BTC loop closure constraints
  const BTCConstraintStats btc_stats =
      AddBTCConstraintsInternal(problem, submaps, config, metrics);
  if (metrics != nullptr) {
    metrics->set_loop_constraint_count(btc_stats.constraints_added);
    metrics->set_btc_frames_without_descriptors(
        btc_stats.frames_without_descriptors);
    metrics->set_btc_frames_without_candidate(
        btc_stats.frames_without_candidate);
    metrics->set_btc_matches_rejected_by_time(
        btc_stats.matches_rejected_by_time);
    metrics->set_btc_matches_rejected_by_gicp(
        btc_stats.matches_rejected_by_gicp);
    metrics->set_btc_accepted_outside_radius(
        btc_stats.accepted_outside_radius);
  }

  ceres::Solver::Options options;
  options.minimizer_progress_to_stdout = true;
  options.max_num_iterations           = config.inner_iteration_num();
  ceres::Solver::Summary summary;

  spdlog::info("Optimization iteration (with GNSS fusion)...");
  ceres::Solve(options, &problem, &summary);
  spdlog::info("{}", summary.FullReport());
  if (metrics != nullptr) {
    metrics->set_initial_cost(summary.initial_cost);
    metrics->set_final_cost(summary.final_cost);
    metrics->set_solver_iterations(static_cast<int>(summary.iterations.size()));
    metrics->set_solver_summary(ceres::TerminationTypeToString(
        summary.termination_type));
    metrics->set_solver_brief_report(summary.BriefReport());
    FillTrajectoryMetrics(timestamped_scan_poses, metrics);
  }

  spdlog::info("GNSS-fused optimization completed successfully");
}

namespace {

BTCConstraintStats AddBTCConstraintsInternal(
    ceres::Problem &problem,
    std::vector<TimestampedPointCloud> &submaps,
    const proto::PgoConfig &config,
    proto::PgoMetrics *metrics) {
  BTCConstraintStats stats;
  if (submaps.size() < 2) {
    spdlog::warn("Not enough submaps for BTC constraint detection: {}", submaps.size());
    return stats;
  }

  spdlog::info("Starting BTC-based loop closure detection on {} submaps", submaps.size());

  const bool visualize_stdescs      = IsEnabledFromEnv("PGO_BTC_VISUALIZE");
  const double median_timestamp_gap = EstimateMedianTimestampGap(submaps);
  const double trajectory_duration =
      std::max(0.0, submaps.back().timestamp - submaps.front().timestamp);
  ConfigSetting btc_cfg = MakeBTCConfig(config, median_timestamp_gap);
  LogBTCConfig(btc_cfg,
               config,
               median_timestamp_gap,
               trajectory_duration,
               visualize_stdescs);

  STDescManager desc_manager(btc_cfg);
  const auto btc_sqrt_information = MakeBTCSqrtInformation(config);

  for (int i = 0; i < static_cast<int>(submaps.size()); ++i) {
    if (!submaps[i].cloud || submaps[i].cloud->empty()) {
      std::vector<STD> empty_stds;
      desc_manager.AddSTDescs(empty_stds);
      continue;
    }

    std::vector<STD> stds;
    desc_manager.GenerateSTDescs(submaps[i].cloud, stds, i);

    if (visualize_stdescs) {
      desc_manager.VisualizeSTDescs(submaps[i].cloud, stds);
    }

    if (stds.empty()) {
      ++stats.frames_without_descriptors;
    }

    TryAddBTCConstraintForFrame(i,
                                stds,
                                problem,
                                submaps,
                                desc_manager,
                                btc_cfg,
                                config,
                                btc_sqrt_information,
                                stats,
                                metrics);

    desc_manager.AddSTDescs(stds);
  }

  spdlog::info(
      "BTC constraint detection finished: {} constraints added, {} frames without descriptors, {} frames without verified candidate, {} matches rejected by time threshold, {} matches rejected by GICP, {} matches rejected by correction sanity gate, {} accepted outside prior radius",
      stats.constraints_added, stats.frames_without_descriptors,
      stats.frames_without_candidate, stats.matches_rejected_by_time,
      stats.matches_rejected_by_gicp,
      stats.matches_rejected_by_sanity,
      stats.accepted_outside_radius);
  return stats;
}

}  // namespace

void AddBTCConstraints(ceres::Problem &problem,
                       std::vector<TimestampedPointCloud> &submaps,
                       const proto::PgoConfig &config) {
  AddBTCConstraintsInternal(problem, submaps, config, nullptr);
}
