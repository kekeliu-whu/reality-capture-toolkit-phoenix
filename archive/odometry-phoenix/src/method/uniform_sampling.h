
#pragma once

#include <pcl/common/common.h>
#include <pcl/filters/filter.h>
#include <pcl/filters/voxel_grid.h>
#include <random>
#include <unordered_map>
#include "common/common_struct.h"
#include "common/math_utils.h"
#include "parameters.h"
namespace lixel
{
struct Leaf
{
  Leaf() : idx_in_input(-1)
  {
  }
  int idx_in_input;
  int idx_in_output;
  std::vector<int> indices_in_input;
};

template <typename PointT>
class UniformSampling : public pcl::Filter<PointT>
{
  typedef typename pcl::Filter<PointT>::PointCloud PointCloud;

  using pcl::Filter<PointT>::filter_name_;
  using pcl::Filter<PointT>::input_;
  using pcl::Filter<PointT>::indices_;
  using pcl::Filter<PointT>::getClassName;

 public:
  typedef std::shared_ptr<UniformSampling<PointT>> Ptr;
  typedef std::shared_ptr<const UniformSampling<PointT>> ConstPtr;

  UniformSampling() = default;
  ~UniformSampling() = default;

  /**
   * Calculate Filter Size According to point_cloud and set as the search_radius_
   * @return filter size
   */
  float calculateRadius(
      const lixel::PointCloudXYZINormal::ConstPtr &point_cloud,
      const DownsampleParam &downsample_param);

  void setRadius(float radius);
  const std::unordered_map<size_t, Leaf> &getIndexMap() const;
  void setRandomSeed(double startTs);

 protected:
  /** \brief The 3D grid leaves. */
  std::unordered_map<size_t, Leaf> index_map_;
  Eigen::Vector4f leaf_size_;
  Eigen::Array4f inverse_leaf_size_;
  Eigen::Vector4i min_b_, max_b_, div_b_, divb_mul_;
  double search_radius_ = 0.0;
  std::mt19937 gen;
  std::uniform_real_distribution<float> dis;
  bool using_random_offset = false;
  /**
   * applyFilter(PointCloud &output) is pure virtual method of pcl::Filter
   * Downsample a Point Cloud using a voxelized grid approach
   * Save the Voxel Relationship in index_map_
   * @param output  the resultant point cloud point message
   */
  void applyFilter(PointCloud &output);
};

}  // namespace lixel
