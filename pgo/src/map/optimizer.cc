#define SOPHUS_DISABLE_ENSURES

#include <ceres/ceres.h>
#include <pcl/kdtree/kdtree.h>
#include <pcl/registration/gicp.h>

#include "factor/gravity_factor.h"
#include "factor/local_parameterization_se3.h"
#include "factor/pose_graph_edge_factor.h"
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
  CHECK(source.get());
  CHECK(target.get());
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
    DLOG(INFO) << "good match: " << score;
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
  int loop_edge_num = 30;  // todo
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

    CHECK_GT(candidate_scan_ids.size(), 0);
    int source_id = candidate_scan_ids[rand() % candidate_scan_ids.size()];

    if (edges_used.find({target_id, source_id}) != edges_used.end() ||
        edges_used.find({source_id, target_id}) != edges_used.end()) {
      continue;
    }

    // match
    Sophus::SE3d T_source_to_target(submaps[target_id].pose.inverse() *
                                    submaps[source_id].pose);

    CHECK_GE(source_id, 0);
    CHECK_LT(source_id, submaps.size());
    CHECK_GE(target_id, 0);
    CHECK_LT(target_id, submaps.size());

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

void Optimize(std::vector<TimestampedPointCloud> &submaps,
              const proto::PgoConfig &config) {
  ceres::Problem problem;

  std::set<ceres::ResidualBlockId> prior_residual_blocks;
  AddParameters(problem, submaps);
  AddAdjacentConstraints(problem, submaps, prior_residual_blocks, config);
  AddGravityConstraits(problem, submaps, prior_residual_blocks, config);

  for (int iter = 0; iter < config.outer_iteration_num(); ++iter) {
    AddLoopClosureConstraints(problem, submaps, config);

    ceres::Solver::Options options;
    options.minimizer_progress_to_stdout = true;
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    DLOG(INFO) << summary.FullReport();

    // clean up non-prior constraints for the next iteration
    RemoveNonPriorConstraints(problem, prior_residual_blocks);
  }
}
