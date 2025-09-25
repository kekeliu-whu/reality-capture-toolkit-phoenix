#pragma once

namespace xcolor {

struct SfmConfig {
  int min_num_matches         = 15;
  bool ignore_watermarks      = false;
  bool refine_focal_length    = true;
  bool refine_principal_point = true;
  bool refine_extra_params    = true;
  double min_tri_angle        = 1.5;

  int outer_opt_num_two_view                           = 1;
  double reproject_error_outlier_thresholds_twoview[1] = {5};    // pixels
  double lidar_error_outlier_thresholds_twoview[1]     = {0.4};  // meters

  int outer_opt_num_multi_view                           = 3;
  double reproject_error_outlier_thresholds_multiview[3] = {5, 4, 3};         // pixels
  double lidar_error_outlier_thresholds_multiview[3]     = {0.4, 0.3, 0.15};  // meters

  double reproject_cauchy_loss_scale = 1.5;  // pixels, half of the smallest reproject_error_outlier_thresholds

  double lidar_cauchy_weight  = 0.1;  // meters, half of the smallest lidar_error_outlier_thresholds
  double lidar_weight_scale   = 0.2;
  bool enable_weight_by_depth = true;

  double pose_prior_rotation_weight    = 0.2 * M_PI / 180.0;
  double pose_prior_translation_weight = 0.05;
  double pose_prior_weight_scale       = 1;

  int ba_optimization_num_threads = 1;

  int min_track_len = 3;

  SfmConfig(int cores_used) { ba_optimization_num_threads = cores_used; }
};

}  // namespace xcolor
