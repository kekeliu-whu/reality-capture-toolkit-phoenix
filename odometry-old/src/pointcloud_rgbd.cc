/*
This code is the implementation of our paper "R3LIVE: A Robust, Real-time, RGB-colored,
LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package".

Author: Jiarong Lin   < ziv.lin.ljr@gmail.com >

If you use any code of this repo in your academic research, please cite at least
one of our papers:
[1] Lin, Jiarong, and Fu Zhang. "R3LIVE: A Robust, Real-time, RGB-colored,
    LiDAR-Inertial-Visual tightly-coupled state Estimation and mapping package."
[2] Xu, Wei, et al. "Fast-lio2: Fast direct lidar-inertial odometry."
[3] Lin, Jiarong, et al. "R2LIVE: A Robust, Real-time, LiDAR-Inertial-Visual
     tightly-coupled state Estimator and mapping."
[4] Xu, Wei, and Fu Zhang. "Fast-lio: A fast, robust lidar-inertial odometry
    package by tightly-coupled iterated kalman filter."
[5] Cai, Yixi, Wei Xu, and Fu Zhang. "ikd-Tree: An Incremental KD Tree for
    Robotic Applications."
[6] Lin, Jiarong, and Fu Zhang. "Loam-livox: A fast, robust, high-precision
    LiDAR odometry and mapping package for LiDARs of small FoV."

For commercial use, please contact me < ziv.lin.ljr@gmail.com > and
Dr. Fu Zhang < fuzhang@hku.hk >.

 Redistribution and use in source and binary forms, with or without
 modification, are permitted provided that the following conditions are met:

 1. Redistributions of source code must retain the above copyright notice,
    this list of conditions and the following disclaimer.
 2. Redistributions in binary form must reproduce the above copyright notice,
    this list of conditions and the following disclaimer in the documentation
    and/or other materials provided with the distribution.
 3. Neither the name of the copyright holder nor the names of its
    contributors may be used to endorse or promote products derived from this
    software without specific prior written permission.

 THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
 LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
 CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
 SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
 INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
 ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
 POSSIBILITY OF SUCH DAMAGE.
*/

#include <spdlog/spdlog.h>

#include "pointcloud_rgbd.h"
#include "image_frame.h"

const double image_obs_cov       = 15;
const double process_noise_sigma = 0.1;

void RGB_pts::set_pos(const vec_3 &pos) {
  m_pos[0] = pos(0);
  m_pos[1] = pos(1);
  m_pos[2] = pos(2);
}

vec_3 RGB_pts::get_pos() { return vec_3(m_pos[0], m_pos[1], m_pos[2]); }

mat_3_3 RGB_pts::get_rgb_cov() {
  mat_3_3 cov_mat = mat_3_3::Zero();
  for (int i = 0; i < 3; i++) {
    cov_mat(i, i) = m_cov_rgb[i];
  }
  return cov_mat;
}

vec_3 RGB_pts::get_rgb() { return vec_3(m_rgb[0], m_rgb[1], m_rgb[2]); }

pcl::PointXYZI RGB_pts::get_pt() {
  pcl::PointXYZI pt;
  pt.x = m_pos[0];
  pt.y = m_pos[1];
  pt.z = m_pos[2];
  return pt;
}

int RGB_pts::update_rgb(const vec_3 &rgb, const double obs_dis, const vec_3 obs_sigma, const double obs_time) {
  if (m_obs_dis != 0 && (obs_dis > m_obs_dis * 1.2)) {
    return 0;
  }

  if (m_N_rgb == 0) {
    // For first time of observation.
    m_last_obs_time = obs_time;
    m_obs_dis       = obs_dis;
    for (int i = 0; i < 3; i++) {
      m_rgb[i]     = rgb[i];
      m_cov_rgb[i] = obs_sigma(i);
    }
    m_N_rgb = 1;
    return 0;
  }
  // State estimation for robotics, section 2.2.6, page 37-38
  for (int i = 0; i < 3; i++) {
    m_cov_rgb[i]     = (m_cov_rgb[i] + process_noise_sigma * (obs_time - m_last_obs_time));  // Add process noise
    double old_sigma = m_cov_rgb[i];
    m_cov_rgb[i]     = sqrt(1.0 / (1.0 / m_cov_rgb[i] / m_cov_rgb[i] + 1.0 / obs_sigma(i) / obs_sigma(i)));
    m_rgb[i] = m_cov_rgb[i] * m_cov_rgb[i] * (m_rgb[i] / old_sigma / old_sigma + rgb(i) / obs_sigma(i) / obs_sigma(i));
  }

  if (obs_dis < m_obs_dis) {
    m_obs_dis = obs_dis;
  }
  m_last_obs_time = obs_time;
  m_N_rgb++;
  return 1;
}

void Global_map::clear() { m_rgb_pts_vec.clear(); }

void Global_map::set_minmum_dis(double minimum_dis) {
  m_hashmap_3d_pts.clear();
  m_minimum_pts_size = minimum_dis;
}

Global_map::Global_map() {
  m_mutex_pts_vec                        = std::make_shared<std::mutex>();
  m_mutex_img_pose_for_projection        = std::make_shared<std::mutex>();
  m_mutex_recent_added_list              = std::make_shared<std::mutex>();
  m_mutex_rgb_pts_in_recent_hitted_boxes = std::make_shared<std::mutex>();
  m_mutex_m_box_recent_hitted            = std::make_shared<std::mutex>();
  m_mutex_pts_last_visited               = std::make_shared<std::mutex>();
  // Allocate memory for pointclouds
  m_rgb_pts_vec.reserve(1e9);
}

Global_map::~Global_map() {};

template int Global_map::append_points_to_global_map<pcl::PointXYZI>(
    pcl::PointCloud<pcl::PointXYZI> &pc_in, double added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec,
    int step);
template int Global_map::append_points_to_global_map<pcl::PointXYZRGB>(
    pcl::PointCloud<pcl::PointXYZRGB> &pc_in, double added_time, std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec,
    int step);

template <typename T>
int Global_map::append_points_to_global_map(pcl::PointCloud<T> &pc_in, double added_time,
                                            std::vector<std::shared_ptr<RGB_pts>> *pts_added_vec, int step) {
  m_in_appending_pts = 1;
  int acc            = 0;
  int rej            = 0;
  if (pts_added_vec != nullptr) {
    pts_added_vec->clear();
  }
  std::unordered_set<std::shared_ptr<RGB_Voxel>> voxels_recent_visited;
  if (m_recent_visited_voxel_activated_time == 0) {
    voxels_recent_visited.clear();
  } else {
    m_mutex_m_box_recent_hitted->lock();
    voxels_recent_visited = m_voxels_recent_visited;
    m_mutex_m_box_recent_hitted->unlock();
    for (Voxel_set_iterator it = voxels_recent_visited.begin(); it != voxels_recent_visited.end();) {
      if (added_time - (*it)->m_last_visited_time > m_recent_visited_voxel_activated_time) {
        it = voxels_recent_visited.erase(it);
        continue;
      }
      it++;
    }
    spdlog::info("Restored voxel number = {}", voxels_recent_visited.size());
  }
  int number_of_voxels_before_add = voxels_recent_visited.size();
  int pt_size                     = pc_in.points.size();
  // step = 4;
  for (int pt_idx = 0; pt_idx < pt_size; pt_idx += step) {
    int add    = 1;
    int grid_x = std::round(pc_in.points[pt_idx].x / m_minimum_pts_size);
    int grid_y = std::round(pc_in.points[pt_idx].y / m_minimum_pts_size);
    int grid_z = std::round(pc_in.points[pt_idx].z / m_minimum_pts_size);
    int box_x  = std::round(pc_in.points[pt_idx].x / m_voxel_resolution);
    int box_y  = std::round(pc_in.points[pt_idx].y / m_voxel_resolution);
    int box_z  = std::round(pc_in.points[pt_idx].z / m_voxel_resolution);
    if (m_hashmap_3d_pts.if_exist(grid_x, grid_y, grid_z)) {
      add = 0;
      if (pts_added_vec != nullptr) {
        pts_added_vec->push_back(m_hashmap_3d_pts.m_map_3d_hash_map[grid_x][grid_y][grid_z]);
      }
    }
    RGB_voxel_ptr box_ptr;
    if (!m_hashmap_voxels.if_exist(box_x, box_y, box_z)) {
      std::shared_ptr<RGB_Voxel> box_rgb = std::make_shared<RGB_Voxel>();
      m_hashmap_voxels.insert(box_x, box_y, box_z, box_rgb);
      box_ptr = box_rgb;
    } else {
      box_ptr = m_hashmap_voxels.m_map_3d_hash_map[box_x][box_y][box_z];
    }
    voxels_recent_visited.insert(box_ptr);
    box_ptr->m_last_visited_time = added_time;
    if (add == 0) {
      rej++;
      continue;
    }
    acc++;
    std::shared_ptr<RGB_pts> pt_rgb = std::make_shared<RGB_pts>();
    pt_rgb->set_pos(vec_3(pc_in.points[pt_idx].x, pc_in.points[pt_idx].y, pc_in.points[pt_idx].z));
    m_rgb_pts_vec.push_back(pt_rgb);
    m_hashmap_3d_pts.insert(grid_x, grid_y, grid_z, pt_rgb);
    box_ptr->add_pt(pt_rgb);
    if (pts_added_vec != nullptr) {
      pts_added_vec->push_back(pt_rgb);
    }
  }
  m_in_appending_pts = 0;
  m_mutex_m_box_recent_hitted->lock();
  m_voxels_recent_visited = voxels_recent_visited;
  m_mutex_m_box_recent_hitted->unlock();
  return (m_voxels_recent_visited.size() - number_of_voxels_before_add);
}

void Global_map::render_pts_in_voxels(std::shared_ptr<Image_frame>          &img_ptr,
                                      std::vector<std::shared_ptr<RGB_pts>> &pts_for_render, double obs_time) {
  double u, v;
  int    hit_count         = 0;
  int    pt_size           = pts_for_render.size();
  m_last_updated_frame_idx = img_ptr->m_frame_idx;
  for (int i = 0; i < pt_size; i++) {
    vec_3 pt_w = pts_for_render[i]->get_pos();
    bool  res  = img_ptr->project_3d_point_in_this_img(pt_w, u, v, nullptr, 1.0);
    if (res == false) {
      continue;
    }
    vec_3 pt_cam = (pt_w - img_ptr->m_pose_w2c_t);
    hit_count++;
    vec_2 gama_bak       = img_ptr->m_gama_para;
    img_ptr->m_gama_para = vec_2(1.0, 0.0);  // Render using normal value?
    double gray          = img_ptr->get_grey_color(u, v, 0);
    vec_3  rgb_color     = img_ptr->get_rgb(u, v, 0);
    pts_for_render[i]->update_rgb(rgb_color, pt_cam.norm(), vec_3(image_obs_cov, image_obs_cov, image_obs_cov),
                                  obs_time);
    img_ptr->m_gama_para = gama_bak;
    // m_rgb_pts_vec[i]->update_rgb( vec_3(gray, gray, gray) );
  }
  // spdlog::info("Render cost time = {}", tim.toc());
  // spdlog::info("Total hit count = {}", hit_count);
}

void Global_map::render_with_a_image(std::shared_ptr<Image_frame> &img_ptr, int if_select) {
  std::vector<std::shared_ptr<RGB_pts>> pts_for_render;
  // pts_for_render = m_rgb_pts_vec;
  if (if_select) {
    selection_points_for_projection(img_ptr, &pts_for_render, nullptr, 1.0);
  } else {
    pts_for_render = m_rgb_pts_vec;
  }
  render_pts_in_voxels(img_ptr, pts_for_render);
}

void Global_map::selection_points_for_projection(std::shared_ptr<Image_frame>          &image_pose,
                                                 std::vector<std::shared_ptr<RGB_pts>> *pc_out_vec,
                                                 std::vector<cv::Point2f> *pc_2d_out_vec, double minimum_dis,
                                                 int skip_step, int use_all_pts) {
  if (pc_out_vec != nullptr) {
    pc_out_vec->clear();
  }
  if (pc_2d_out_vec != nullptr) {
    pc_2d_out_vec->clear();
  }
  Hash_map_2d<int, int>   mask_index;
  Hash_map_2d<int, float> mask_depth;

  std::map<int, cv::Point2f> map_idx_draw_center;
  std::map<int, cv::Point2f> map_idx_draw_center_raw_pose;

  int    u, v;
  double u_f, v_f;
  // cv::namedWindow("Mask", cv::WINDOW_FREERATIO);
  int acc     = 0;
  int blk_rej = 0;
  // int pts_size = m_rgb_pts_vec.size();
  std::vector<std::shared_ptr<RGB_pts>> pts_for_projection;
  m_mutex_m_box_recent_hitted->lock();
  std::unordered_set<std::shared_ptr<RGB_Voxel>> boxes_recent_hitted = m_voxels_recent_visited;
  m_mutex_m_box_recent_hitted->unlock();
  if ((!use_all_pts) && boxes_recent_hitted.size()) {
    m_mutex_rgb_pts_in_recent_hitted_boxes->lock();

    for (Voxel_set_iterator it = boxes_recent_hitted.begin(); it != boxes_recent_hitted.end(); it++) {
      // pts_for_projection.push_back( (*it)->m_pts_in_grid.back() );
      if ((*it)->m_pts_in_grid.size()) {
        pts_for_projection.push_back((*it)->m_pts_in_grid.back());
        // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ 0 ] );
        // pts_for_projection.push_back( ( *it )->m_pts_in_grid[ ( *it )->m_pts_in_grid.size()-1 ] );
      }
    }

    m_mutex_rgb_pts_in_recent_hitted_boxes->unlock();
  } else {
    pts_for_projection = m_rgb_pts_vec;
  }
  int pts_size = pts_for_projection.size();
  for (int pt_idx = 0; pt_idx < pts_size; pt_idx += skip_step) {
    vec_3  pt    = pts_for_projection[pt_idx]->get_pos();
    double depth = (pt - image_pose->m_pose_w2c_t).norm();
    if (depth > m_maximum_depth_for_projection) {
      continue;
    }
    if (depth < m_minimum_depth_for_projection) {
      continue;
    }
    bool res = image_pose->project_3d_point_in_this_img(pt, u_f, v_f, nullptr, 1.0);
    if (res == false) {
      continue;
    }
    u = std::round(u_f / minimum_dis) * minimum_dis;  // Why can not work
    v = std::round(v_f / minimum_dis) * minimum_dis;
    if ((!mask_depth.if_exist(u, v)) || mask_depth.m_map_2d_hash_map[u][v] > depth) {
      acc++;
      if (mask_index.if_exist(u, v)) {
        // erase old point
        int old_idx = mask_index.m_map_2d_hash_map[u][v];
        blk_rej++;
        map_idx_draw_center.erase(map_idx_draw_center.find(old_idx));
        map_idx_draw_center_raw_pose.erase(map_idx_draw_center_raw_pose.find(old_idx));
      }
      mask_index.m_map_2d_hash_map[u][v]   = (int)pt_idx;
      mask_depth.m_map_2d_hash_map[u][v]   = (float)depth;
      map_idx_draw_center[pt_idx]          = cv::Point2f(v, u);
      map_idx_draw_center_raw_pose[pt_idx] = cv::Point2f(u_f, v_f);
    }
  }

  if (pc_out_vec != nullptr) {
    for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++) {
      // pc_out_vec->push_back(m_rgb_pts_vec[it->first]);
      pc_out_vec->push_back(pts_for_projection[it->first]);
    }
  }

  if (pc_2d_out_vec != nullptr) {
    for (auto it = map_idx_draw_center.begin(); it != map_idx_draw_center.end(); it++) {
      pc_2d_out_vec->push_back(map_idx_draw_center_raw_pose[it->first]);
    }
  }
}

void Global_map::save_to_pcd(const std::string &file_name, int save_pts_with_views) {
  spdlog::info("Save Rgb points to {}", file_name);
  fflush(stdout);
  pcl::PointCloud<pcl::PointXYZRGB> pc_rgb;
  long                              pt_size = m_rgb_pts_vec.size();
  pc_rgb.resize(pt_size);
  long pt_count = 0;
  spdlog::info("Saving offline map... ");
  for (long i = pt_size - 1; i > 0; i--)
  // for (int i = 0; i  <  pt_size; i++)
  {
    if (m_rgb_pts_vec[i]->m_N_rgb < save_pts_with_views) {
      continue;
    }
    pcl::PointXYZRGB pt;
    pc_rgb.points[pt_count].x = m_rgb_pts_vec[i]->m_pos[0];
    pc_rgb.points[pt_count].y = m_rgb_pts_vec[i]->m_pos[1];
    pc_rgb.points[pt_count].z = m_rgb_pts_vec[i]->m_pos[2];
    pc_rgb.points[pt_count].r = m_rgb_pts_vec[i]->m_rgb[2];
    pc_rgb.points[pt_count].g = m_rgb_pts_vec[i]->m_rgb[1];
    pc_rgb.points[pt_count].b = m_rgb_pts_vec[i]->m_rgb[0];
    pt_count++;
  }
  spdlog::info("Saving offline map 100% ...");
  pc_rgb.resize(pt_count);
  spdlog::info("Total have {} points.", pt_count);
  spdlog::info("Now write to: {}", file_name);
  pcl::io::savePCDFileBinary(file_name, pc_rgb);
}
