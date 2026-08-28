#pragma once

#include <cstdint>

extern "C" {

int navvis_recon_slam_submap_rotation(
    const double* node_quaternion_xyzw, const double* gravity_observation,
    double* output_quaternion_xyzw);

void* navvis_recon_slam_octree_create(std::uint64_t point_count,
                                      const float* points);
void navvis_recon_slam_octree_destroy(void* octree);
int navvis_recon_slam_octree_nearest(
    void* octree, std::uint64_t query_count, const float* queries,
    float search_radius, std::int32_t num_threads, std::uint8_t* found,
    std::uint64_t* indices, float* distances_squared);

int navvis_recon_slam_range_centroid_filter(
    std::uint64_t input_count, const float* origins, const float* points,
    float resolution, std::uint64_t output_capacity,
    std::uint64_t* output_count, float* output_origins, float* output_points);

int navvis_recon_slam_label_surfel_cells(
    std::uint64_t previous_count, const std::int64_t* previous_keys,
    std::uint64_t point_count, const float* points, double grid_origin_x,
    double grid_origin_y, double grid_origin_z, double inverse_resolution,
    std::uint64_t output_capacity, std::uint64_t* output_count,
    std::int64_t* output_keys, std::uint64_t* point_labels);

int navvis_recon_slam_update_split_surfels(
    std::uint64_t state_count, float* primary_weights,
    std::uint32_t* primary_counts, float* primary_means,
    float* primary_covariances, float* primary_viewpoints,
    float* secondary_weights, std::uint32_t* secondary_counts,
    float* secondary_means, float* secondary_covariances,
    float* secondary_viewpoints, std::uint8_t* is_split,
    float* split_normals, std::uint8_t* primary_dirty,
    std::uint8_t* secondary_dirty, std::uint64_t point_count,
    const std::uint64_t* labels, const float* origins, const float* points,
    std::uint8_t maintain_surfels);

int navvis_recon_slam_maintain_split_surfels(
    std::uint64_t state_count, const float* primary_weights,
    const std::uint32_t* primary_counts, const float* primary_means,
    const float* primary_covariances, const float* primary_viewpoints,
    const std::uint32_t* secondary_counts, const float* secondary_means,
    const float* secondary_covariances, const float* secondary_viewpoints,
    std::uint8_t* is_split, float* split_normals,
    std::uint8_t* primary_dirty, std::uint8_t* secondary_dirty);

int navvis_recon_slam_merge_surfels(
    std::uint64_t state_count, const std::int64_t* keys,
    float* primary_weights, std::uint32_t* primary_counts,
    float* primary_means, float* primary_covariances,
    float* primary_viewpoints, float* secondary_weights,
    std::uint32_t* secondary_counts, float* secondary_means,
    float* secondary_covariances, float* secondary_viewpoints,
    std::uint8_t* is_split, float* split_normals,
    std::uint8_t* primary_dirty, std::uint8_t* secondary_dirty,
    double grid_origin_x, double grid_origin_y, double grid_origin_z,
    double inverse_resolution, std::uint64_t source_count,
    const std::uint64_t* sources, std::uint64_t* merge_count);

int navvis_recon_slam_oriented_surfel_geometry(
    std::uint64_t state_count, const float* covariances, const float* means,
    const float* viewpoint_means, float* normals, float* eigenvalues);

void* navvis_recon_slam_probability_grid_create(float resolution);
void navvis_recon_slam_probability_grid_destroy(void* grid);
std::uint64_t navvis_recon_slam_probability_grid_size(const void* grid);
int navvis_recon_slam_probability_grid_export(
    const void* grid, std::uint64_t capacity, std::uint64_t* output_count,
    std::int32_t* indices, std::uint16_t* values);
int navvis_recon_slam_probability_grid_load(
    void* grid, std::uint64_t cell_count, const std::int32_t* indices,
    const std::uint16_t* values);
int navvis_recon_slam_probability_grid_insert(
    void* grid, std::uint64_t point_count, const float* points,
    const float* origins);

int navvis_recon_slam_inverse_pose(const double* translation,
                                   const double* quaternion_xyzw,
                                   double* output_pose);
int navvis_recon_slam_compose_pose(const double* lhs_translation,
                                   const double* lhs_quaternion_xyzw,
                                   const double* rhs_translation,
                                   const double* rhs_quaternion_xyzw,
                                   double* output_pose);
int navvis_recon_slam_compose_pose_normalized(
    const double* lhs_translation, const double* lhs_quaternion_xyzw,
    const double* rhs_translation, const double* rhs_quaternion_xyzw,
    double* output_pose);
int navvis_recon_slam_icp_normalization_pose(
    const double* translation, const double* quaternion_xyzw,
    double* output_pose);
int navvis_recon_slam_transform_points_double_matrix_cast(
    std::uint64_t point_count, const float* input_points,
    const double* translation, const double* quaternion_xyzw,
    float* output_points);
int navvis_recon_slam_transform_points_raw_float_quaternion(
    std::uint64_t point_count, const float* input_points,
    const double* translation, const double* quaternion_xyzw,
    float* output_points);
int navvis_recon_slam_deskew_points(
    std::uint64_t point_count, const double* input_points,
    const double* quaternion_xyzw, const double* translation,
    float* output_points);
int navvis_recon_slam_transform_submap_range_data(
    std::uint64_t point_count, const float* input_points,
    const float* input_normals, const float* input_origins,
    const double* node_translation, const double* node_quaternion_xyzw,
    const double* submap_translation, const double* submap_quaternion_xyzw,
    float* output_points, float* output_normals, float* output_origins,
    double* output_relative_pose);

int navvis_recon_slam_point_plane_step(
    std::uint64_t correspondence_count, const float* source_points,
    const float* target_points, const float* target_normals,
    const double* normalization_translation,
    const double* normalization_quaternion_xyzw, double* output_translation,
    double* output_quaternion_xyzw, double* output_delta,
    double* output_scale, double* output_normal_matrix,
    double* output_right_hand_side);

}  // extern "C"
