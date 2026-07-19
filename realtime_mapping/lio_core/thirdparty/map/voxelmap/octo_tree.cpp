#include "octo_tree.hpp"

int static plane_id = 0;

namespace xvmp
{
void OctoTree::init_plane(const std::vector<pointStamped> &points, Plane::Ptr plane)
{
  if (points.size() < 5)
    return;

  double max_points_timestamp = 0;

  // 初始化：如果存在双面墙点，直接返回失败
  // for (size_t i = 1; i < points.size(); i++)
  // {
  //   pointStamped p0 = points[0];
  //   pointStamped pi = points[i];
  //   if (p0.view_vec.dot(pi.view_vec) < -0.99 || p0.normal.dot(pi.normal) < 0)
  //   {
  //     return;
  //   }
  // }

  plane->covariance = Eigen::Matrix3d::Zero();
  plane->center = Eigen::Vector3d::Zero();
  plane->normal = Eigen::Vector3d::Zero();
  plane->points_size = points.size();

  // 计算平面基本参数
  Eigen::Vector3d mean_view_vec(0, 0, 0);
  for (auto pv : points)
  {
    plane->covariance += pv.point * pv.point.transpose();
    plane->center += pv.point;
    mean_view_vec += pv.view_vec;
    max_points_timestamp = std::max(max_points_timestamp, pv.timestamp);
  }
  plane->center = plane->center / plane->points_size;
  plane->covariance =
      plane->covariance / plane->points_size - plane->center * plane->center.transpose();
  mean_view_vec = mean_view_vec / plane->points_size;

  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(plane->covariance);
  Eigen::Vector3d evals = eig.eigenvalues();
  Eigen::Matrix3d evecs = eig.eigenvectors();
  double planarity = fabs(evals(2)) > 1e-8 ? (evals(1) - evals(0)) / evals(2) : 0;

  plane->eigen_vectors = evecs;
  plane->normal = evecs.col(0);
  plane->normal = plane->normal / plane->normal.norm();
  plane->d = -plane->normal.dot(plane->center);
  if (plane->normal.dot(mean_view_vec) < 0)
  {
    plane->normal *= -1.0;
  }

  plane->d = -plane->normal.dot(plane->center);
  plane->min_eigen_value = evals(0);
  plane->mid_eigen_value = evals(1);
  plane->max_eigen_value = evals(2);
  plane->planarity = planarity;
  plane->std_deviation = sqrt(plane->min_eigen_value);
  plane->layer = layer_;
  plane->radius = 2.5 * sqrt(plane->max_eigen_value);
  plane->timestamp = max_points_timestamp;

  if (plane->std_deviation < planer_threshold_ && planarity > 0.2 && plane->radius > 0.05)
  {
    // 平面性校验通过
    plane->is_plane = true;
    if (plane->last_update_points_size == 0)
    {
      plane->last_update_points_size = plane->points_size;
      plane->is_update = true;
    }
    else if (plane->points_size - plane->last_update_points_size > 100)
    {
      plane->last_update_points_size = plane->points_size;
      plane->is_update = true;
    }

    if (!plane->is_init)
    {
      plane->id = plane_id;
      plane_id++;
      plane->is_init = true;
    }
  }
  else
  {
    // 平面校验未通过

    // 未初始化的先初始化
    if (!plane->is_init)
    {
      plane->id = plane_id;
      plane_id++;
      plane->is_init = true;
    }

    if (plane->last_update_points_size == 0)
    {
      // 从未被更新过，置为可更新
      plane->last_update_points_size = plane->points_size;
      plane->is_update = true;
    }
    else if (plane->points_size - plane->last_update_points_size > 100)
    {
      // 上次更新到现在新增点超过100， 置为可更新
      plane->last_update_points_size = plane->points_size;
      plane->is_update = true;
    }
  }
}

void OctoTree::update_plane(const std::vector<pointStamped> &points, Plane::Ptr plane)
{
  // 更新：如果存在双面墙点，直接返回失败
  // for (size_t i = 1; i < points.size(); i++)
  // {
  //   pointStamped p0 = points[0];
  //   pointStamped pi = points[i];
  //   if (p0.view_vec.dot(pi.view_vec) < -0.99 || p0.normal.dot(pi.normal) < 0)
  //   {
  //     return;
  //   }
  // }

  double max_points_timestamp = 0;

  // 重新计算平面参数
  Eigen::Matrix3d old_covariance = plane->covariance;
  Eigen::Vector3d old_center = plane->center;
  Eigen::Matrix3d sum_ppt =
      (plane->covariance + plane->center * plane->center.transpose()) * plane->points_size;
  Eigen::Vector3d sum_p = plane->center * plane->points_size;
  Eigen::Vector3d mean_view_vec(0, 0, 0);
  for (size_t i = 0; i < points.size(); i++)
  {
    Eigen::Vector3d pv = points[i].point;
    sum_ppt += pv * pv.transpose();
    sum_p += pv;
    mean_view_vec += points[i].view_vec;
    max_points_timestamp = std::max(max_points_timestamp, points[i].timestamp);
  }
  mean_view_vec = mean_view_vec / plane->points_size;
  plane->points_size = plane->points_size + points.size();
  plane->center = sum_p / plane->points_size;
  plane->covariance = sum_ppt / plane->points_size - plane->center * plane->center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eig(plane->covariance);
  Eigen::Vector3d evals = eig.eigenvalues();
  Eigen::Matrix3d evecs = eig.eigenvectors();
  double planarity = fabs(evals(2)) > 1e-8 ? (evals(1) - evals(0)) / evals(2) : 0;

  plane->eigen_vectors = evecs;
  plane->normal = evecs.col(0);
  plane->normal = plane->normal / plane->normal.norm();
  if (plane->normal.dot(mean_view_vec) < 0)
  {
    plane->normal *= -1.0;
  }

  plane->d = -plane->normal.dot(plane->center);
  plane->min_eigen_value = evals(0);
  plane->mid_eigen_value = evals(1);
  plane->max_eigen_value = evals(2);
  plane->planarity = planarity;
  plane->std_deviation = sqrt(plane->min_eigen_value);
  plane->layer = layer_;
  plane->radius = 2.5 * sqrt(plane->max_eigen_value);
  plane->timestamp = max_points_timestamp;

  if (plane->std_deviation < planer_threshold_ && planarity > 0.2 && plane->radius > 0.05)
  {
    // 是平面，可更新
    plane->is_plane = true;
    plane->is_update = true;
  }
  else
  {
    // 不是平面，可更新
    plane->is_plane = false;
    plane->is_update = true;
  }
}

void OctoTree::init_octo_tree()
{
  // 必须点数足够才能初始化
  if (temp_points_.size() > min_plane_update_threshold_)
  {
    init_plane(temp_points_, active_plane_ptr_);
    if (active_plane_ptr_->is_plane == true)
    {
      // 找到了平面，八叉树状态fix
      octo_state_ = OCTREE_FIXED;

      // 点数已达阈值，八叉树不可更新，平面不可重新计算
      if (temp_points_.size() > max_points_size_)
      {
        plane_reinit_enable_ = false;
        octree_update_enable_ = false;
        std::vector<pointStamped>().swap(temp_points_);
      }
    }
    else
    {
      // 未找到平面，在八叉树子节点继续找
      octo_state_ = OCTREE_MIDDLE;
      cut_octo_tree();
    }
    new_points_num_ = 0;
    std::vector<pointStamped>().swap(temp_points_);

    last_visit_time_ = std::max(last_visit_time_, active_plane_ptr_->timestamp);
  }
}

void OctoTree::cut_octo_tree()
{
  // 达到最小的尺度，返回
  if (layer_ >= max_layer_)
  {
    octo_state_ = OCTREE_FIXED;
    return;
  }

  // 把点分到每个子节点
  for (size_t i = 0; i < temp_points_.size(); i++)
  {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].point[0] > voxel_center_[0])
    {
      xyz[0] = 1;
    }
    if (temp_points_[i].point[1] > voxel_center_[1])
    {
      xyz[1] = 1;
    }
    if (temp_points_[i].point[2] > voxel_center_[2])
    {
      xyz[2] = 1;
    }

    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr)
    {
      leaves_[leafnum] = new OctoTree(
          max_layer_,
          layer_ + 1,
          layer_point_size_,
          max_points_size_,
          max_cov_points_size_,
          planer_threshold_);
      leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
      leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
      leaves_[leafnum]->quater_length_ = quater_length_ / 2;
    }

    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_num_++;
  }

  // 在每个子节点上寻找平面，找不到则在更深一层的子节点上寻找
  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
    {
      if (leaves_[i]->temp_points_.size() > leaves_[i]->min_plane_update_threshold_)
      {
        init_plane(leaves_[i]->temp_points_, leaves_[i]->active_plane_ptr_);
        if (leaves_[i]->active_plane_ptr_->is_plane)
        {
          leaves_[i]->octo_state_ = OCTREE_FIXED;
        }
        else
        {
          leaves_[i]->octo_state_ = OCTREE_MIDDLE;
          leaves_[i]->cut_octo_tree();
        }
        leaves_[i]->new_points_num_ = 0;
        std::vector<pointStamped>().swap(leaves_[i]->temp_points_);
      }

      leaves_[i]->last_visit_time_ =
          std::max(leaves_[i]->last_visit_time_, leaves_[i]->active_plane_ptr_->timestamp);
    }
  }
}

void OctoTree::reset_octo_tree()
{
  octo_state_ = OCTREE_UNINITIALIZED;
  new_points_num_ = 0;
  all_points_num_ = 0;
  octree_update_enable_ = true;
  plane_reinit_enable_ = false;

  active_plane_ptr_.reset(new Plane);
  // recent_plane_ptr_.reset(new Plane);

  std::vector<pointStamped>().swap(temp_points_);
  std::vector<pointStamped>().swap(new_points_);
  std::vector<pointStamped>().swap(recent_points_);

  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
      leaves_[i]->reset_octo_tree();
  }
}

void OctoTree::clear_points()
{
  std::vector<pointStamped>().swap(temp_points_);
  std::vector<pointStamped>().swap(new_points_);
  std::vector<pointStamped>().swap(recent_points_);

  new_points_num_ = 0;
  all_points_num_ = 0;

  for (uint i = 0; i < 8; i++)
  {
    if (leaves_[i] != nullptr)
      leaves_[i]->clear_points();
  }
}

void OctoTree::UpdateOctoTree(pointStamped &pv)
{
  if (OCTREE_UNINITIALIZED == octo_state_)
  {
    // 初始化octree
    new_points_num_++;
    all_points_num_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > min_plane_update_threshold_)
    {
      init_octo_tree();
    }
  }
  else
  {
    if (active_plane_ptr_->is_plane)
    {
      if (pv.view_vec.dot(active_plane_ptr_->normal) < 0)
      {
        reset_octo_tree();
      }

      if (octree_update_enable_)
      {
        new_points_num_++;
        all_points_num_++;
        if (plane_reinit_enable_)
        {
          temp_points_.push_back(pv);
        }
        else
        {
          new_points_.push_back(pv);
        }
        if (new_points_num_ > update_size_threshold_)
        {
          if (plane_reinit_enable_)
          {
            init_plane(temp_points_, active_plane_ptr_);
          }
          new_points_num_ = 0;
        }
        if (all_points_num_ >= max_cov_points_size_)
        {
          plane_reinit_enable_ = false;
          std::vector<pointStamped>().swap(temp_points_);
        }
        if (all_points_num_ >= max_points_size_)
        {
          octree_update_enable_ = false;
          active_plane_ptr_->update_enable = false;
          std::vector<pointStamped>().swap(new_points_);
        }
      }
      else
      {
        return;
      }
    }
    else
    {
      // recent点未组成平面
      // 未达八叉树最深层，清空buffer，更新子节点
      if (layer_ < max_layer_)
      {
        if (temp_points_.size() != 0)
        {
          std::vector<pointStamped>().swap(temp_points_);
        }
        if (new_points_.size() != 0)
        {
          std::vector<pointStamped>().swap(new_points_);
        }
        int xyz[3] = {0, 0, 0};
        if (pv.point[0] > voxel_center_[0])
        {
          xyz[0] = 1;
        }
        if (pv.point[1] > voxel_center_[1])
        {
          xyz[1] = 1;
        }
        if (pv.point[2] > voxel_center_[2])
        {
          xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr)
        {
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
        else
        {
          leaves_[leafnum] = new OctoTree(
              max_layer_,
              layer_ + 1,
              layer_point_size_,
              max_points_size_,
              max_cov_points_size_,
              planer_threshold_);
          leaves_[leafnum]->layer_point_size_ = layer_point_size_;
          leaves_[leafnum]->voxel_center_[0] = voxel_center_[0] + (2 * xyz[0] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[1] = voxel_center_[1] + (2 * xyz[1] - 1) * quater_length_;
          leaves_[leafnum]->voxel_center_[2] = voxel_center_[2] + (2 * xyz[2] - 1) * quater_length_;
          leaves_[leafnum]->quater_length_ = quater_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      }
      else
      {
        // 已达到八叉树最深层，继续尝试初始化或更新平面
        if (octree_update_enable_)
        {
          new_points_num_++;
          all_points_num_++;
          if (plane_reinit_enable_)
          {
            temp_points_.push_back(pv);
          }
          else
          {
            new_points_.push_back(pv);
          }
          if (new_points_num_ > update_size_threshold_)
          {
            if (plane_reinit_enable_)
            {
              init_plane(temp_points_, active_plane_ptr_);
            }
            else
            {
              update_plane(new_points_, active_plane_ptr_);
              new_points_.clear();
            }
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_cov_points_size_)
          {
            plane_reinit_enable_ = false;
            std::vector<pointStamped>().swap(temp_points_);
          }
          if (all_points_num_ >= max_points_size_)
          {
            octree_update_enable_ = false;
            active_plane_ptr_->update_enable = false;
            std::vector<pointStamped>().swap(new_points_);
          }
        }
      }
    }
  }
}

// 检查平面是否存在，存在则返回true和target plane
bool OctoTree::find_plane(const Plane::Ptr plane_ptr, Plane::Ptr &target_plane)
{
  if (PLANE_TYPE_IDENTITY == compare_plane(plane_ptr, active_plane_ptr_))
  {
    target_plane = active_plane_ptr_;
    return true;
  }

  for (auto &old_plane_ptr : plane_vec_)
  {
    if (PLANE_TYPE_IDENTITY == compare_plane(plane_ptr, old_plane_ptr))
    {
      target_plane = old_plane_ptr;
      return true;
    }
  }

  return false;
}

int OctoTree::compare_plane(const Plane::Ptr plane0_ptr, const Plane::Ptr plane1_ptr)
{
  if (!plane0_ptr->is_plane || !plane1_ptr->is_plane)
    return PLANE_TYPE_INVALID;

  int plane_type = 0;  // identity, parallel, negative, orthorhombic,others, not_plane

  // check normal
  double n = plane0_ptr->normal.dot(plane1_ptr->normal);
  if (n < 0.0)
    plane_type = PLANE_TYPE_NEGATIVE;
  else if (fabs(n) < 0.1)
    plane_type = PLANE_TYPE_ORTHORHOMBIC;
  else if (n > 0.99)
    plane_type = PLANE_TYPE_PARALLEL;
  else
    plane_type = PLANE_TYPE_OTHERS;

  // check plane distance
  double d0 = fabs(plane0_ptr->normal.dot(plane1_ptr->center) + plane0_ptr->d);
  double d1 = fabs(plane1_ptr->normal.dot(plane0_ptr->center) + plane1_ptr->d);
  double d = (d0 + d1) / 2;
  if (PLANE_TYPE_PARALLEL == plane_type && d < 0.05)
    plane_type = PLANE_TYPE_IDENTITY;

  return plane_type;
}

bool OctoTree::insert_plane(Plane::Ptr plane_ptr)
{
  for (auto &old_plane_ptr : plane_vec_)
  {
    if (PLANE_TYPE_IDENTITY == compare_plane(plane_ptr, old_plane_ptr))
    {
      return false;
    }
  }

  plane_vec_.push_back(plane_ptr);
  return true;
}

Plane::Ptr OctoTree::get_active_plane()
{
  if (active_plane_ptr_->is_plane)
    return active_plane_ptr_;
  else
    return NULL;
}

void OctoTree::get_all_planes(std::vector<Plane::Ptr> &planes)
{
  if (active_plane_ptr_->is_plane)
    planes.push_back(active_plane_ptr_);

  planes.insert(planes.end(), plane_vec_.begin(), plane_vec_.end());
  for (auto &leaf : leaves_)
  {
    if (leaf)
    {
      leaf->get_all_planes(planes);
    }
  }
}

size_t OctoTree::get_all_points_count()
{
  size_t n_tmp = temp_points_.size();
  size_t n_new = new_points_.size();
  size_t n_recent = recent_points_.size();

  size_t n = n_tmp + n_new + n_recent;

  for (auto &leaf : leaves_)
  {
    if (leaf)
    {
      n += leaf->get_all_points_count();
    }
  }

  return n;
}

void OctoTree::get_planes_in_current_layer(
    std::vector<Plane::Ptr> &planes,
    const Eigen::Vector3d &view_vec,
    bool use_normal,
    double plane_threlshold)
{
  if (planarity_and_normal_check(active_plane_ptr_, view_vec, use_normal, plane_threlshold))
  {
    planes.push_back(active_plane_ptr_);
  }

  for (auto &plane : plane_vec_)
  {
    if (planarity_and_normal_check(plane, view_vec, use_normal, plane_threlshold))
      planes.push_back(plane);
  }
}

bool OctoTree::search_plane(
    const Eigen::Vector3d &pt,
    Plane::Ptr &plane,
    const Eigen::Vector3d &view_vec,
    bool use_normal,
    double plane_threlshold)
{
  std::vector<Plane::Ptr> planes;
  get_planes_in_current_layer(planes, view_vec, use_normal, plane_threlshold);

  if (planes.empty())
  {
    int xyz[3] = {0, 0, 0};
    if (pt[0] > voxel_center_[0])
    {
      xyz[0] = 1;
    }
    if (pt[1] > voxel_center_[1])
    {
      xyz[1] = 1;
    }
    if (pt[2] > voxel_center_[2])
    {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] != nullptr)
    {
      Plane::Ptr leaf_plane;
      if (leaves_[leafnum]->search_plane(pt, leaf_plane, view_vec, use_normal, plane_threlshold))
      {
        planes.push_back(leaf_plane);
      }

      // std::vector<Plane::Ptr> child_planes;
      // leaves_[leafnum]->get_planes_in_current_layer(
      //     child_planes, view_vec, use_normal, plane_threlshold);
      // planes.insert(planes.end(), child_planes.begin(), child_planes.end());
    }
  }

  // distance check
  for (auto &pl : planes)
  {
    double dist_to_center = (pt - pl->center).norm();
    double dist_to_plane = fabs(pt.dot(pl->normal) + pl->d);
    double range_dist = sqrt(dist_to_center * dist_to_center - dist_to_plane * dist_to_plane);
    if (range_dist < pl->radius)
    {
      plane = pl;
      return true;
    }
  }

  plane = NULL;
  return false;
}

void OctoTree::get_active_planes(std::vector<Plane::Ptr> &planes)
{
  // if (active_plane_ptr_->is_plane)
  //     planes.push_back(active_plane_ptr_);

  // for (auto &leaf : leaves_)
  // {
  //     if (leaf)
  //     {
  //         leaf->get_active_planes(planes);
  //     }
  // }
}

bool OctoTree::planarity_and_normal_check(
    const Plane::Ptr plane,
    const Eigen::Vector3d &view_vec,
    bool use_normal,
    double plane_threlshold)
{
  bool valid_flag = false;
  if (plane->is_plane && plane->std_deviation < plane_threlshold)
  {
    if (!use_normal)
      valid_flag = true;
    else if (plane->normal.dot(view_vec) > 0)
      valid_flag = true;
  }

  return valid_flag;
}

}  // namespace xvmp