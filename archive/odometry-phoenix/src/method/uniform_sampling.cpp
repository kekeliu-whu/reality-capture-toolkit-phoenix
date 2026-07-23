//
// Created by youyuan on 24-2-19.
//
#include "uniform_sampling.h"

namespace lixel
{
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////
template <>
void UniformSampling<PointXYZINormal>::applyFilter(PointCloud& output)
{
  // Has the input dataset been set already?
  if (!input_)
  {
    PCL_WARN("[pcl::%s::detectKeypoints] No input dataset given!\n", getClassName().c_str());
    output.width = output.height = 0;
    output.points.clear();
    return;
  }
  output.height = 1;       // downsampling breaks the organized structure
  output.is_dense = true;  // we filter out invalid points

  /*** Add offset to origin pcl ***/
  V3F offset(dis(gen), dis(gen), dis(gen));
  std::cout << "offset:" << offset.transpose() << std::endl;
  PointCloudXYZINormal::Ptr offset_pcl(new PointCloudXYZINormal);
  PointCloudXYZINormal::ConstPtr using_pcl = input_;
  if (using_random_offset)
  {
    offset_pcl->height = 1;
    offset_pcl->width = input_->width;
    offset_pcl->is_dense = true;
    for (const PointXYZINormal& p : input_->points)
    {
      PointXYZINormal offset_point;
      offset_point.x = p.x + offset.x();
      offset_point.y = p.y + offset.y();
      offset_point.z = p.z + offset.z();
      offset_pcl->push_back(offset_point);
    }
    using_pcl = offset_pcl;
  }
  Eigen::Vector4f min_p, max_p;

  /*** Calculate Bounding ***/
  pcl::getMinMax3D<PointXYZINormal>(*using_pcl, min_p, max_p);
  min_b_[0] = static_cast<int>(floor(min_p[0] * inverse_leaf_size_[0]));
  max_b_[0] = static_cast<int>(floor(max_p[0] * inverse_leaf_size_[0]));
  min_b_[1] = static_cast<int>(floor(min_p[1] * inverse_leaf_size_[1]));
  max_b_[1] = static_cast<int>(floor(max_p[1] * inverse_leaf_size_[1]));
  min_b_[2] = static_cast<int>(floor(min_p[2] * inverse_leaf_size_[2]));
  max_b_[2] = static_cast<int>(floor(max_p[2] * inverse_leaf_size_[2]));

  /*** Compute Spatial Hash Param ***/
  index_map_.clear();
  div_b_ = max_b_ - min_b_ + Eigen::Vector4i::Ones();
  div_b_[3] = 0;
  divb_mul_ = Eigen::Vector4i(1, div_b_[0], div_b_[0] * div_b_[1], 0);

  /*** build a set of leaves with the point index closest to the leaf center ***/
  for (size_t cp = 0; cp < indices_->size(); ++cp)
  {
    if (!using_pcl->is_dense)
      if (!std::isfinite(using_pcl->points[(*indices_)[cp]].x) || !std::isfinite(using_pcl->points[(*indices_)[cp]].y) ||
          !std::isfinite(using_pcl->points[(*indices_)[cp]].z))
        continue;

    /*** Calculate Int Spatial Hash Index ***/
    Eigen::Vector4i ijk = Eigen::Vector4i::Zero();
    ijk[0] = static_cast<int>(floor(using_pcl->points[(*indices_)[cp]].x * inverse_leaf_size_[0]));
    ijk[1] = static_cast<int>(floor(using_pcl->points[(*indices_)[cp]].y * inverse_leaf_size_[1]));
    ijk[2] = static_cast<int>(floor(using_pcl->points[(*indices_)[cp]].z * inverse_leaf_size_[2]));

    int idx = (ijk - min_b_).dot(divb_mul_);
    Leaf& leaf = index_map_[idx];
    leaf.indices_in_input.push_back((*indices_)[cp]);
    if (leaf.idx_in_input == -1)
    {
      leaf.idx_in_input = (*indices_)[cp];
      continue;
    }

    /*** Check to see if this point is closer to the leaf center than the previous one we saved ***/
    Eigen::Vector3f center;
    center[0] = ijk[0] * leaf_size_[0] + leaf_size_[0] / 2;
    center[1] = ijk[1] * leaf_size_[1] + leaf_size_[1] / 2;
    center[2] = ijk[2] * leaf_size_[2] + leaf_size_[2] / 2;
    float diff_cur = (using_pcl->points[(*indices_)[cp]].getVector3fMap() - center).squaredNorm();
    float diff_prev = (using_pcl->points[leaf.idx_in_input].getVector3fMap() - center).squaredNorm();
    if (diff_cur < diff_prev)
      leaf.idx_in_input = (*indices_)[cp];
  }

  /*** Go over all leaves and copy data from input pcl ***/
  output.points.resize(index_map_.size());
  int cp = 0;
  for (auto& it : index_map_)
  {
    it.second.idx_in_output = cp;
    output.points[cp++] = input_->points[it.second.idx_in_input];
  }

  output.width = static_cast<uint32_t>(output.points.size());
}

template <>
float UniformSampling<PointXYZINormal>::calculateRadius(
    const PointCloudXYZINormal::ConstPtr& point_cloud,
    const DownsampleParam& downsample_param)
{
  /** 1. First DownSample to hundreds points **/
  pcl::VoxelGrid<PointXYZINormal> voxel_filter;
  PointCloudXYZINormal::Ptr init_downsample(new PointCloudXYZINormal);

  const float& init_dis = downsample_param.init_pca_downsample_dis;
  voxel_filter.setLeafSize(init_dis, init_dis, init_dis);
  voxel_filter.setInputCloud(point_cloud);
  voxel_filter.filter(*init_downsample);

  /** 2. SVD decomposition calculation **/
  auto& points_vec = init_downsample->points;
  int point_num = points_vec.size();
  std::vector<V3F> data;
  data.reserve(point_num);
  for (int i = 0; i < point_num; i++)
  {
    V3F pv(points_vec[i].x, points_vec[i].y, points_vec[i].z);
    data.push_back(pv);
  }
  V3F sigma;
  M3F V;
  SVD(data, V, sigma);
  //
  float S = 0;
  if (downsample_param.area_method == lixel::SufaceAreaMethod::ellipsoid)
  {
    S = 4.0 * PI / 3.0 * (sigma(0) * sigma(2) + sigma(1) * sigma(2) + sigma(0) * sigma(1)) / float(point_num);
  }
  else if (downsample_param.area_method == lixel::SufaceAreaMethod::cube_4_side)
  {
    S = 2.0 * (sigma(0) * sigma(2) + sigma(1) * sigma(2)) / float(point_num);
  }
  float d0 = sqrt(S / float(downsample_param.ref_downsample_point_num));
  /** 3. calculate final downsample size **/
  PointCloudXYZINormal::Ptr first_downsample(new PointCloudXYZINormal);
  voxel_filter.setLeafSize(d0, d0, d0);
  voxel_filter.setInputCloud(point_cloud);
  voxel_filter.filter(*first_downsample);
  float d = sqrt(float(first_downsample->size()) / float(downsample_param.ref_downsample_point_num)) * d0;
  if (d > downsample_param.max_downsample_dis)
    d = downsample_param.max_downsample_dis;
  else if (d < downsample_param.base_downsample_dis)
    d = downsample_param.base_downsample_dis;

  return d;
}

template <>
void UniformSampling<PointXYZINormal>::setRadius(float radius)
{
  dis.param(std::uniform_real_distribution<float>::param_type(-radius / 2, radius / 2));
  leaf_size_[0] = leaf_size_[1] = leaf_size_[2] = static_cast<float>(radius);
  // Avoid division errors
  if (leaf_size_[3] == 0)
    leaf_size_[3] = 1;
  // Use multiplications instead of divisions
  inverse_leaf_size_ = Eigen::Array4f::Ones() / leaf_size_.array();
  search_radius_ = radius;
}
template <>
void UniformSampling<PointXYZINormal>::setRandomSeed(double startTs)
{
  gen.seed(std::floor(startTs));
  using_random_offset = true;
}

template <>
const std::unordered_map<size_t, Leaf>& UniformSampling<PointXYZINormal>::getIndexMap() const
{
  return index_map_;
}
}  // namespace lixel