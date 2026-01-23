#include "voxel_map_util.h"

namespace {

int g_plane_id = 0;

double g_min_plane_likeness;

}  // namespace

OctoTree::OctoTree(int max_layer, int layer, std::vector<int> layer_point_size,
                   int max_point_size, int max_cov_points_size, float planer_threshold)
    : max_layer_(max_layer), layer_(layer), layer_point_size_(layer_point_size), max_points_size_(max_point_size), max_cov_points_size_(max_cov_points_size), planer_threshold_(planer_threshold) {
  temp_points_.clear();
  octo_state_     = 0;
  new_points_num_ = 0;
  all_points_num_ = 0;
  // when new points num > 5, do a update
  update_size_threshold_      = 5;
  init_octo_                  = false;
  update_enable_              = true;
  update_cov_enable_          = true;
  max_plane_update_threshold_ = layer_point_size_[layer_];
  for (int i = 0; i < 8; i++) {
    leaves_[i] = nullptr;
  }
  plane_ptr_ = new Plane;
}

// check is plane , calc plane parameters including plane covariance
void OctoTree::InitPlane(const std::vector<pointWithCovMeta> &points, Plane *plane) {
  plane->plane_cov   = Eigen::Matrix<double, 6, 6>::Zero();
  plane->covariance  = Eigen::Matrix3d::Zero();
  plane->center      = Eigen::Vector3d::Zero();
  plane->normal      = Eigen::Vector3d::Zero();
  plane->points_size = points.size();
  plane->radius      = 0;
  for (auto pv : points) {
    plane->covariance += pv.pw * pv.pw.transpose();
    plane->center += pv.pw;
  }
  plane->center     = plane->center / plane->points_size;
  plane->covariance = plane->covariance / plane->points_size -
                      plane->center * plane->center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
  Eigen::Matrix3d evecs           = es.eigenvectors().real();
  Eigen::Vector3d evals           = es.eigenvalues().real();
  Eigen::Matrix3f::Index evalsMin = 0, evalsMid = 1, evalsMax = 2;  // SelfAdjointEigenSolver's eigen values are in increase order
  // plane covariance calculation
  Eigen::Matrix3d J_Q;
  J_Q << 1.0 / plane->points_size, 0, 0, 0, 1.0 / plane->points_size, 0, 0, 0,
      1.0 / plane->points_size;
  double plane_likeness = 2 * (evals(evalsMid) - evals(evalsMin)) / evals.sum();
  if (evals(evalsMin) < planer_threshold_ && plane_likeness > g_min_plane_likeness) {
    for (int i = 0; i < points.size(); i++) {
      Eigen::Matrix<double, 6, 3> J;
      Eigen::Matrix3d F;
      for (int m = 0; m < 3; m++) {
        if (m != (int)evalsMin) {
          Eigen::Matrix<double, 1, 3> F_m =
              (points[i].pw - plane->center).transpose() /
              ((plane->points_size) * (evals[evalsMin] - evals[m])) *
              (evecs.col(m) * evecs.col(evalsMin).transpose() +
               evecs.col(evalsMin) * evecs.col(m).transpose());
          F.row(m) = F_m;
        } else {
          Eigen::Matrix<double, 1, 3> F_m;
          F_m << 0, 0, 0;
          F.row(m) = F_m;
        }
      }
      J.block<3, 3>(0, 0) = evecs * F;
      J.block<3, 3>(3, 0) = J_Q;
      plane->plane_cov += J * points[i].cov * J.transpose();
    }

    plane->is_plane = true;
  } else {
    plane->is_plane = false;
  }

  plane->normal          = evecs.col(evalsMin);
  plane->y_normal        = evecs.col(evalsMid);
  plane->x_normal        = evecs.col(evalsMax);
  plane->min_eigen_value = evals(evalsMin);
  plane->mid_eigen_value = evals(evalsMid);
  plane->max_eigen_value = evals(evalsMax);
  plane->radius          = sqrt(evals(evalsMax));
  plane->d               = -plane->normal.dot(plane->center);

  plane->id = g_plane_id++;
}

// only updaye plane normal, center and radius with new points
void OctoTree::UpdatePlane(const std::vector<pointWithCovMeta> &points, Plane *plane) {
  Eigen::Matrix3d old_covariance = plane->covariance;
  Eigen::Vector3d old_center     = plane->center;
  Eigen::Matrix3d sum_ppt =
      (plane->covariance + plane->center * plane->center.transpose()) *
      plane->points_size;
  Eigen::Vector3d sum_p = plane->center * plane->points_size;
  for (size_t i = 0; i < points.size(); i++) {
    Eigen::Vector3d pv = points[i].pw;
    sum_ppt += pv * pv.transpose();
    sum_p += pv;
  }
  plane->points_size = plane->points_size + points.size();
  plane->center      = sum_p / plane->points_size;
  plane->covariance  = sum_ppt / plane->points_size -
                      plane->center * plane->center.transpose();
  Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> es(plane->covariance);
  Eigen::Matrix3d evecs           = es.eigenvectors().real();
  Eigen::Vector3d evals           = es.eigenvalues().real();
  Eigen::Matrix3f::Index evalsMin = 0, evalsMid = 1, evalsMax = 2;

  plane->normal          = evecs.col(evalsMin);
  plane->y_normal        = evecs.col(evalsMid);
  plane->x_normal        = evecs.col(evalsMax);
  plane->min_eigen_value = evals(evalsMin);
  plane->mid_eigen_value = evals(evalsMid);
  plane->max_eigen_value = evals(evalsMax);
  plane->radius          = sqrt(evals(evalsMax));
  plane->d               = -plane->normal.dot(plane->center);

  if (evals(evalsMin) < planer_threshold_) {
    plane->is_plane = true;
  } else {
    plane->is_plane = false;
  }
}

void OctoTree::InitOctoTree() {
  if (temp_points_.size() > max_plane_update_threshold_) {
    InitPlane(temp_points_, plane_ptr_);
    if (plane_ptr_->is_plane == true) {
      octo_state_ = 0;
      if (temp_points_.size() > max_cov_points_size_) {
        update_cov_enable_ = false;
      }
      if (temp_points_.size() > max_points_size_) {
        update_enable_ = false;
      }
    } else {
      octo_state_ = 1;
      CutOctoTree();
    }
    init_octo_      = true;
    new_points_num_ = 0;
    //      temp_points_.clear();
  }
}

void OctoTree::CutOctoTree() {
  if (layer_ >= max_layer_) {
    octo_state_ = 0;
    return;
  }
  for (size_t i = 0; i < temp_points_.size(); i++) {
    int xyz[3] = {0, 0, 0};
    if (temp_points_[i].pw[0] > voxel_center_[0]) {
      xyz[0] = 1;
    }
    if (temp_points_[i].pw[1] > voxel_center_[1]) {
      xyz[1] = 1;
    }
    if (temp_points_[i].pw[2] > voxel_center_[2]) {
      xyz[2] = 1;
    }
    int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
    if (leaves_[leafnum] == nullptr) {
      leaves_[leafnum] = new OctoTree(
          max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
          max_cov_points_size_, planer_threshold_);
      leaves_[leafnum]->voxel_center_[0] =
          voxel_center_[0] + (2 * xyz[0] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[1] =
          voxel_center_[1] + (2 * xyz[1] - 1) * quarter_length_;
      leaves_[leafnum]->voxel_center_[2] =
          voxel_center_[2] + (2 * xyz[2] - 1) * quarter_length_;
      leaves_[leafnum]->quarter_length_ = quarter_length_ / 2;
    }
    leaves_[leafnum]->temp_points_.push_back(temp_points_[i]);
    leaves_[leafnum]->new_points_num_++;
  }
  for (uint32_t i = 0; i < 8; i++) {
    if (leaves_[i] != nullptr) {
      if (leaves_[i]->temp_points_.size() >
          leaves_[i]->max_plane_update_threshold_) {
        InitPlane(leaves_[i]->temp_points_, leaves_[i]->plane_ptr_);
        if (leaves_[i]->plane_ptr_->is_plane) {
          leaves_[i]->octo_state_ = 0;
        } else {
          leaves_[i]->octo_state_ = 1;
          leaves_[i]->CutOctoTree();
        }
        leaves_[i]->init_octo_      = true;
        leaves_[i]->new_points_num_ = 0;
      }
    }
  }
}

void OctoTree::UpdateOctoTree(const pointWithCovMeta &pv) {
  if (!init_octo_) {
    // If voxel not initialized yet, try to initialize voxel as plane
    new_points_num_++;
    all_points_num_++;
    temp_points_.push_back(pv);
    if (temp_points_.size() > max_plane_update_threshold_) {
      InitOctoTree();
    }
  } else {
    if (plane_ptr_->is_plane) {
      // If already initialized and voxel is plane, try to re-initialize plane every N frames
      if (update_enable_) {
        new_points_num_++;
        all_points_num_++;
        if (update_cov_enable_) {
          temp_points_.push_back(pv);
        } else {
          new_points_.push_back(pv);
        }
        if (new_points_num_ > update_size_threshold_) {
          if (update_cov_enable_) {
            InitPlane(temp_points_, plane_ptr_);
          }
          new_points_num_ = 0;
        }
        if (all_points_num_ >= max_cov_points_size_) {
          update_cov_enable_ = false;
          std::vector<pointWithCovMeta>().swap(temp_points_);
        }
        if (all_points_num_ >= max_points_size_) {
          update_enable_ = false;
          std::vector<pointWithCovMeta>().swap(new_points_);
        }
      } else {
        return;
      }
    } else {
      // If already initialized but voxel is not plane
      //   If not at max layer, delete all temp points of current node (so all nodes that fail initialization first time are no longer considered plane), and try to update next layer node
      //   If already at max layer, try to re-initialize plane every N frames
      if (layer_ < max_layer_) {
        if (temp_points_.size() != 0) {
          std::vector<pointWithCovMeta>().swap(temp_points_);
        }
        if (new_points_.size() != 0) {
          std::vector<pointWithCovMeta>().swap(new_points_);
        }
        int xyz[3] = {0, 0, 0};
        if (pv.pw[0] > voxel_center_[0]) {
          xyz[0] = 1;
        }
        if (pv.pw[1] > voxel_center_[1]) {
          xyz[1] = 1;
        }
        if (pv.pw[2] > voxel_center_[2]) {
          xyz[2] = 1;
        }
        int leafnum = 4 * xyz[0] + 2 * xyz[1] + xyz[2];
        if (leaves_[leafnum] != nullptr) {
          leaves_[leafnum]->UpdateOctoTree(pv);
        } else {
          leaves_[leafnum] = new OctoTree(
              max_layer_, layer_ + 1, layer_point_size_, max_points_size_,
              max_cov_points_size_, planer_threshold_);
          leaves_[leafnum]->layer_point_size_ = layer_point_size_;
          leaves_[leafnum]->voxel_center_[0] =
              voxel_center_[0] + (2 * xyz[0] - 1) * quarter_length_;
          leaves_[leafnum]->voxel_center_[1] =
              voxel_center_[1] + (2 * xyz[1] - 1) * quarter_length_;
          leaves_[leafnum]->voxel_center_[2] =
              voxel_center_[2] + (2 * xyz[2] - 1) * quarter_length_;
          leaves_[leafnum]->quarter_length_ = quarter_length_ / 2;
          leaves_[leafnum]->UpdateOctoTree(pv);
        }
      } else {
        if (update_enable_) {
          new_points_num_++;
          all_points_num_++;
          if (update_cov_enable_) {
            temp_points_.push_back(pv);
          } else {
            new_points_.push_back(pv);
          }
          if (new_points_num_ > update_size_threshold_) {
            if (update_cov_enable_) {
              InitPlane(temp_points_, plane_ptr_);
            } else {
              UpdatePlane(new_points_, plane_ptr_);
              new_points_.clear();
            }
            new_points_num_ = 0;
          }
          if (all_points_num_ >= max_cov_points_size_) {
            update_cov_enable_ = false;
            std::vector<pointWithCovMeta>().swap(temp_points_);
          }
          if (all_points_num_ >= max_points_size_) {
            update_enable_ = false;
            std::vector<pointWithCovMeta>().swap(new_points_);
          }
        }
      }
    }
  }
}

// void mapJet(double v, double vmin, double vmax, uint8_t &r, uint8_t &g,
//             uint8_t &b) {
//   r = 255;
//   g = 255;
//   b = 255;

//   if (v < vmin) {
//     v = vmin;
//   }

//   if (v > vmax) {
//     v = vmax;
//   }

//   double dr, dg, db;

//   if (v < 0.1242) {
//     db = 0.504 + ((1. - 0.504) / 0.1242) * v;
//     dg = dr = 0.;
//   } else if (v < 0.3747) {
//     db = 1.;
//     dr = 0.;
//     dg = (v - 0.1242) * (1. / (0.3747 - 0.1242));
//   } else if (v < 0.6253) {
//     db = (0.6253 - v) * (1. / (0.6253 - 0.3747));
//     dg = 1.;
//     dr = (v - 0.3747) * (1. / (0.6253 - 0.3747));
//   } else if (v < 0.8758) {
//     db = 0.;
//     dr = 1.;
//     dg = (0.8758 - v) * (1. / (0.8758 - 0.6253));
//   } else {
//     db = 0.;
//     dg = 0.;
//     dr = 1. - (v - 0.8758) * ((1. - 0.504) / (1. - 0.8758));
//   }

//   r = (uint8_t)(255 * dr);
//   g = (uint8_t)(255 * dg);
//   b = (uint8_t)(255 * db);
// }

void BuildVoxelMap(const std::vector<pointWithCov> &input_points,
                   const float voxel_size, const int max_layer,
                   const std::vector<int> &layer_point_size,
                   const int max_points_size, const int max_cov_points_size,
                   const float planer_threshold,
                   std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  uint32_t plsize = input_points.size();
  for (uint32_t i = 0; i < plsize; i++) {
    const pointWithCovMeta p_v = input_points[i];
    float loc_xyz[3];
    // 1. Compute voxel location of the point
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.pw[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                      (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    // 2. Insert point into corresponding voxel
    if (iter != feat_map.end()) {
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position]                   = octo_tree;
      feat_map[position]->quarter_length_  = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->temp_points_.push_back(p_v);
      feat_map[position]->new_points_num_++;
      feat_map[position]->layer_point_size_ = layer_point_size;
    }
  }
  // 3. Initialize all voxels
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    iter->second->InitOctoTree();
  }
}

void UpdateVoxelMap(const std::vector<pointWithCov> &input_points,
                    const float voxel_size, const int max_layer,
                    const std::vector<int> &layer_point_size,
                    const int max_points_size, const int max_cov_points_size,
                    const float planer_threshold,
                    std::unordered_map<VoxelLoc, OctoTree *> &feat_map) {
  uint32_t plsize = input_points.size();
  for (uint32_t i = 0; i < plsize; i++) {
    const pointWithCovMeta p_v = input_points[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_v.pw[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                      (int64_t)loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      feat_map[position]->UpdateOctoTree(p_v);
    } else {
      OctoTree *octo_tree =
          new OctoTree(max_layer, 0, layer_point_size, max_points_size,
                       max_cov_points_size, planer_threshold);
      feat_map[position]                   = octo_tree;
      feat_map[position]->quarter_length_  = voxel_size / 4;
      feat_map[position]->voxel_center_[0] = (0.5 + position.x) * voxel_size;
      feat_map[position]->voxel_center_[1] = (0.5 + position.y) * voxel_size;
      feat_map[position]->voxel_center_[2] = (0.5 + position.z) * voxel_size;
      feat_map[position]->UpdateOctoTree(p_v);
    }
  }
}

void BuildSingleResidual(const pointWithCov &pv, const OctoTree *current_octo,
                         const int current_layer, const int max_layer,
                         const double sigma_num, bool &is_success,
                         double &prob, ptpl &single_ptpl) {
  double radius_k     = 3;
  Eigen::Vector3d p_w = pv.pw;
  if (current_octo->plane_ptr_->is_plane) {
    Plane &plane                      = *current_octo->plane_ptr_;
    Eigen::Vector3d p_world_to_center = p_w - plane.center;
    float dis_to_plane                = fabs(plane.normal.dot(p_w) + plane.d);
    float dis_to_center               = p_world_to_center.squaredNorm();
    float range_dis                   = sqrt(dis_to_center - dis_to_plane * dis_to_plane);

    if (range_dis <= radius_k * plane.radius) {
      Eigen::Matrix<double, 1, 6> J_nq;
      J_nq.block<1, 3>(0, 0) = p_world_to_center;
      J_nq.block<1, 3>(0, 3) = -plane.normal;
      double sigma_l         = J_nq * plane.plane_cov * J_nq.transpose();
      sigma_l += plane.normal.transpose() * pv.cov * plane.normal;
      if (dis_to_plane < sigma_num * sqrt(sigma_l)) {
        is_success       = true;
        double this_prob = 1.0 / (sqrt(sigma_l)) *
                           exp(-0.5 * dis_to_plane * dis_to_plane / sigma_l);
        if (this_prob > prob) {
          prob                  = this_prob;
          single_ptpl.pv        = pv;
          single_ptpl.plane_cov = plane.plane_cov;
          single_ptpl.normal    = plane.normal;
          single_ptpl.center    = plane.center;
          single_ptpl.d         = plane.d;
          single_ptpl.layer     = current_layer;
        }
        return;
      } else {
        // is_success = false;
        return;
      }
    } else {
      // is_success = false;
      return;
    }
  } else {
    if (current_layer < max_layer) {
      for (size_t leafnum = 0; leafnum < 8; leafnum++) {
        if (current_octo->leaves_[leafnum] != nullptr) {
          OctoTree *leaf_octo = current_octo->leaves_[leafnum];
          BuildSingleResidual(pv, leaf_octo, current_layer + 1, max_layer,
                              sigma_num, is_success, prob, single_ptpl);
        }
      }
      return;
    } else {
      // is_success = false;
      return;
    }
  }
}

// void GetUpdatePlane(const OctoTree *current_octo, const int pub_max_voxel_layer,
//                     std::vector<Plane> &plane_list) {
//   if (current_octo->layer_ > pub_max_voxel_layer) {
//     return;
//   }
//   if (current_octo->plane_ptr_->is_update) {
//     plane_list.push_back(*current_octo->plane_ptr_);
//   }
//   if (current_octo->layer_ < current_octo->max_layer_) {
//     if (!current_octo->plane_ptr_->is_plane) {
//       for (size_t i = 0; i < 8; i++) {
//         if (current_octo->leaves_[i] != nullptr) {
//           GetUpdatePlane(current_octo->leaves_[i], pub_max_voxel_layer,
//                          plane_list);
//         }
//       }
//     }
//   }
//   return;
// }

// void BuildResidualListTBB(const unordered_map<VOXEL_LOC, OctoTree *>
// &voxel_map,
//                           const double voxel_size, const double sigma_num,
//                           const int max_layer,
//                           const std::vector<pointWithCovMeta> &pv_list,
//                           std::vector<ptpl> &ptpl_list,
//                           std::vector<Eigen::Vector3d> &non_match) {
//   std::mutex mylock;
//   ptpl_list.clear();
//   std::vector<ptpl> all_ptpl_list(pv_list.size());
//   std::vector<bool> useful_ptpl(pv_list.size());
//   std::vector<size_t> index(pv_list.size());
//   for (size_t i = 0; i < index.size(); ++i) {
//     index[i] = i;
//     useful_ptpl[i] = false;
//   }
//   std::for_each(
//       std::execution::par_unseq, index.begin(), index.end(),
//       [&](const size_t &i) {
//         pointWithCovMeta pv = pv_list[i];
//         float loc_xyz[3];
//         for (int j = 0; j < 3; j++) {
//           loc_xyz[j] = pv.point_world[j] / voxel_size;
//           if (loc_xyz[j] < 0) {
//             loc_xyz[j] -= 1.0;
//           }
//         }
//         VOXEL_LOC position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
//                            (int64_t)loc_xyz[2]);
//         auto iter = voxel_map.find(position);
//         if (iter != voxel_map.end()) {
//           OctoTree *current_octo = iter->second;
//           ptpl single_ptpl;
//           bool is_success = false;
//           double prob = 0;
//           build_single_residual(pv, current_octo, 0, max_layer, sigma_num,
//                                 is_success, prob, single_ptpl);
//           if (!is_success) {
//             VOXEL_LOC near_position = position;
//             if (loc_xyz[0] > (current_octo->voxel_center_[0] +
//                               current_octo->quater_length_)) {
//               near_position.x = near_position.x + 1;
//             } else if (loc_xyz[0] < (current_octo->voxel_center_[0] -
//                                      current_octo->quater_length_)) {
//               near_position.x = near_position.x - 1;
//             }
//             if (loc_xyz[1] > (current_octo->voxel_center_[1] +
//                               current_octo->quater_length_)) {
//               near_position.y = near_position.y + 1;
//             } else if (loc_xyz[1] < (current_octo->voxel_center_[1] -
//                                      current_octo->quater_length_)) {
//               near_position.y = near_position.y - 1;
//             }
//             if (loc_xyz[2] > (current_octo->voxel_center_[2] +
//                               current_octo->quater_length_)) {
//               near_position.z = near_position.z + 1;
//             } else if (loc_xyz[2] < (current_octo->voxel_center_[2] -
//                                      current_octo->quater_length_)) {
//               near_position.z = near_position.z - 1;
//             }
//             auto iter_near = voxel_map.find(near_position);
//             if (iter_near != voxel_map.end()) {
//               build_single_residual(pv, iter_near->second, 0, max_layer,
//                                     sigma_num, is_success, prob, single_ptpl);
//             }
//           }
//           if (is_success) {

//             mylock.lock();
//             useful_ptpl[i] = true;
//             all_ptpl_list[i] = single_ptpl;
//             mylock.unlock();
//           } else {
//             mylock.lock();
//             useful_ptpl[i] = false;
//             mylock.unlock();
//           }
//         }
//       });
//   for (size_t i = 0; i < useful_ptpl.size(); i++) {
//     if (useful_ptpl[i]) {
//       ptpl_list.push_back(all_ptpl_list[i]);
//     }
//   }
// }

void BuildResidualListOMP(const unordered_map<VoxelLoc, OctoTree *> &voxel_map,
                          const double voxel_size, const double sigma_num,
                          const int max_layer,
                          const std::vector<pointWithCov> &pv_list,
                          std::vector<ptpl> &ptpl_list,
                          std::vector<Eigen::Vector3d> &non_match) {
  ptpl_list.clear();
  std::vector<ptpl> all_ptpl_list(pv_list.size());
  std::vector<int> useful_ptpl(pv_list.size(), false);
#pragma omp parallel for
  for (int i = 0; i < pv_list.size(); i++) {
    pointWithCov pv = pv_list[i];
    float loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = pv.pw[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1.0;
      }
    }
    VoxelLoc position((int64_t)loc_xyz[0], (int64_t)loc_xyz[1],
                      (int64_t)loc_xyz[2]);
    ptpl single_ptpl;
    bool is_success = false;
    double prob     = 0;
    for (int dx = -1; dx <= 1; ++dx) {
      for (int dy = -1; dy <= 1; ++dy) {
        for (int dz = -1; dz <= 1; ++dz) {
          VoxelLoc position_new(position.x + dx, position.y + dy, position.z + dz);
          auto iter = voxel_map.find(position_new);
          if (iter != voxel_map.end()) {
            OctoTree *current_octo = iter->second;
            BuildSingleResidual(pv, current_octo, 0, max_layer, sigma_num,
                                is_success, prob, single_ptpl);
          }
        }
      }
    }
    if (is_success) {
      useful_ptpl[i]   = true;
      all_ptpl_list[i] = single_ptpl;
    } else {
      useful_ptpl[i] = false;
    }
  }
  for (size_t i = 0; i < useful_ptpl.size(); i++) {
    if (useful_ptpl[i]) {
      ptpl_list.push_back(all_ptpl_list[i]);
    }
  }
}

// eq.1
void CalcBodyCov(Eigen::Vector3d &pb, const float range_inc,
                 const float degree_inc, Eigen::Matrix3d &cov) {
  float range     = sqrt(pb[0] * pb[0] + pb[1] * pb[1] + pb[2] * pb[2]);
  float range_var = range_inc * range_inc;
  Eigen::Matrix2d direction_var;
  direction_var << pow(sin(DEG2RAD(degree_inc)), 2), 0, 0,
      pow(sin(DEG2RAD(degree_inc)), 2);
  Eigen::Vector3d direction(pb);
  // if z=0, error will occur in calcBodyCov. to be solved
  if (direction.z() == 0) {
    direction.z() = 0.001;
  }
  direction.normalize();
  Eigen::Matrix3d direction_hat;
  direction_hat << 0, -direction(2), direction(1), direction(2), 0,
      -direction(0), -direction(1), direction(0), 0;
  Eigen::Vector3d base_vector1(1, 1,
                               -(direction(0) + direction(1)) / direction(2));
  base_vector1.normalize();
  Eigen::Vector3d base_vector2 = base_vector1.cross(direction);
  base_vector2.normalize();
  Eigen::Matrix<double, 3, 2> N;
  N << base_vector1(0), base_vector2(0), base_vector1(1), base_vector2(1),
      base_vector1(2), base_vector2(2);
  Eigen::Matrix<double, 3, 2> A = range * direction_hat * N;
  cov                           = direction * range_var * direction.transpose() +
        A * direction_var * A.transpose();
};

struct M_POINT {
  Eigen::Vector3d center;
  int count = 0;
  std::vector<Eigen::Vector3d> points;
};

template <typename T>
void DownSamplingVoxel(const pcl::PointCloud<PointType> &cloud_in,
                       pcl::PointCloud<PointType> &cloud_out,
                       double voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  unordered_map<VoxelLoc, M_POINT> feat_map;

  for (uint32_t i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    int loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.center += p_c;
      iter->second.count++;
    } else {
      M_POINT p;
      p.center           = p_c;
      p.count            = 1;
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  uint32_t i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.center.cast<float>() / iter->second.count;
    i++;
  }
}

template <typename T>
void DownSamplingVoxelRandom(const pcl::PointCloud<PointType> &cloud_in,
                             pcl::PointCloud<PointType> &cloud_out,
                             double voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  unordered_map<VoxelLoc, M_POINT> feat_map;

  for (uint32_t i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().cast<double>();
    int loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.points.push_back(p_c);
    } else {
      M_POINT p;
      p.points.push_back(p_c);
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  uint32_t i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.points.at(rand() % iter->second.points.size()).cast<float>();
    i++;
  }
}

template void DownSamplingVoxel<PointType>(const pcl::PointCloud<PointType> &cloud_in,
                                           pcl::PointCloud<PointType> &cloud_out,
                                           double voxel_size);

template void DownSamplingVoxelRandom<PointType>(const pcl::PointCloud<PointType> &cloud_in,
                                                 pcl::PointCloud<PointType> &cloud_out,
                                                 double voxel_size);

void InitVoxelMapParams(double min_plane_likeness) {
  g_min_plane_likeness = min_plane_likeness;
}
