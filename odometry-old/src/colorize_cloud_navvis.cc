#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <Eigen/Eigen>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/format.hpp>
#include <opencv2/opencv.hpp>

#include "common_lib.h"
#include "pointcloud_rgbd.hpp"

struct TimeStampedState {
  double time;
  Eigen::Quaterniond rot;
  Eigen::Vector3d pos;
};

std::vector<TimeStampedState> ReadLidarStates(const std::string &filename) {
  std::vector<TimeStampedState> states;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    spdlog::error("infile.is_open() failed");
    exit(1);
  }

  std::string line;
  std::getline(infile, line);

  TimeStampedState state;
  while (infile >> state.time >> state.pos[0] >> state.pos[1] >> state.pos[2] >> state.rot.x() >> state.rot.y() >> state.rot.z() >> state.rot.w()) {
    state.rot.normalize();
    states.push_back(state);
  }
  spdlog::info("Number of poses loaded: {}", states.size());
  return states;
}

std::vector<double> ReadTriggerTimestamps(const std::string &filename) {
  std::vector<double> timestamps;
  std::ifstream infile(filename);
  if (!infile.is_open()) {
    spdlog::error("infile.is_open() failed");
    exit(1);
  }

  TimeStampedState state;
  uint64_t timestamp_ns;
  while (infile >> timestamp_ns) {
    timestamps.push_back(timestamp_ns * 1e-9);
  }
  spdlog::info("Number of trigger timestamps loaded: {}", timestamps.size());
  return timestamps;
}

struct ImageInfo {
  TimeStampedState state;
  bool exist = false;
};

// get T_cw
std::deque<ImageInfo> ComputeTriggerCameraPoses(const std::vector<TimeStampedState> &states, const std::vector<double> &trigger_timestamps) {
  // cam0
  Eigen::Quaterniond rot1{0.55503322346987594,
                          -0.43084031432705472,
                          -0.42751705014452018,
                          -0.5688092089892105};
  Eigen::Vector3d pos1{0.0015287756742656672,
                       0.11306746601651824,
                       -0.00036282575151124277};

  // lidar
  Eigen::Quaterniond rot2{0.89572341110116316,
                          0.00093473423644999923,
                          -0.00016557714642211738,
                          0.4446106944973463};
  Eigen::Vector3d pos2{-0.00079115205805683781,
                       0.00011542889283456398,
                       0.048006061318188097};

  Eigen::Quaterniond rot_cl = rot1.conjugate() * rot2;
  Eigen::Vector3d pos_cl    = rot1.conjugate() * (pos2 - pos1);

  std::deque<ImageInfo> retv;

  for (auto &t : trigger_timestamps) {
    auto it = std::lower_bound(states.begin(), states.end(), t, [](const TimeStampedState &lhs, double rhs) {
      return lhs.time < rhs;
    });
    --it;

    if (it == states.begin() - 1 || it == states.end() - 1) {
      ImageInfo ii;
      ii.exist = false;
      retv.push_back(ii);
    } else {
      double s = (t - it->time) / ((it + 1)->time - it->time);
      CHECK_GE(s, 0);
      CHECK_LE(s, 1);

      Eigen::Vector3d pos_wl    = (1 - s) * it->pos + s * (it + 1)->pos;
      Eigen::Quaterniond rot_wl = it->rot.slerp(s, (it + s)->rot);

      ImageInfo ii;
      ii.state.time = t;
      ii.state.rot  = rot_cl * rot_wl.conjugate();
      ii.state.pos  = rot_cl * (rot_wl.conjugate() * -pos_wl) + pos_cl;
      ii.exist      = true;
      retv.push_back(ii);
    }
  }

  return retv;
}

int main() {
  auto lidar_states          = ReadLidarStates(std::string(ROOT_DIR) + "/PCD/traj.txt");
  auto camera_trigger_places = ReadTriggerTimestamps("/home/rick/Documents/raw_data/navvis/vlx/2022-07-28_03.43.39/colorize-kk/trigger_list.txt");
  auto camera_states         = ComputeTriggerCameraPoses(lidar_states, camera_trigger_places);

  pcl::PointCloud<pcl::PointXYZRGB> cloud_rgb;
  std::deque<pcl::PointCloud<pcl::PointXYZI>> scans;
  for (int i = 500; i < lidar_states.size(); ++i) {
    spdlog::info("{}", i);

    pcl::PointCloud<pcl::PointXYZI> scan;
    pcl::io::loadPCDFile(std::string(ROOT_DIR) + "/PCD/scans/" + to_string(i) + ".pcd", scan);

    for (auto &p : scan) {
      pcl::PointXYZRGB pt;
      pt.getVector3fMap() = (lidar_states[i].rot * p.getVector3fMap().cast<double>() + lidar_states[i].pos).cast<float>();
      p.getVector3fMap() = pt.getVector3fMap();
      cloud_rgb.push_back(pt);
    }

    scan.header.stamp = uint64_t(lidar_states[i].time * 1e6);
    scans.push_back(scan);
  }
  pcl::io::savePCDFileBinary(std::string(ROOT_DIR) + "/PCD/cloud_full.pcd", cloud_rgb);

  // for (int i = 0; i < camera_states.size(); ++i) {
  //   if (!camera_states[i].exist) {
  //     continue;
  //   }

  //   auto filename = (boost::format("/home/rick/Documents/raw_data/navvis/vlx/2022-07-28_03.43.39/colorize-kk/cam-export/%05d-cam0.jpg") % i).str();
  //   auto img      = cv::imread(filename);

  //   if (img.empty()) {
  //     LOG(INFO) << "image not found: " << filename;
  //     continue;
  //   }

    // cv::flip(img, img, 0);

    // Eigen::Matrix3d K;
    // K << 910.6, 0, 2732,
    //     0, 910.6, 1820,
    //     0, 0, 1;

    // for (auto &p : cloud_rgb) {
    //   p.r = p.g = p.b = 0;

    //   Eigen::Vector3d p_c = K * (camera_states[i].state.rot * p.getVector3fMap().cast<double>() + camera_states[i].state.pos);
    //   if (p_c[2] <= 0) {
    //     continue;
    //   }

    //   p_c /= p_c[2];

    //   if (p_c[0] >= 0 && p_c[0] < img.cols && p_c[1] >= 0 && p_c[1] < img.rows) {
    //     auto pixel = img.at<cv::Vec3b>(p_c[1], p_c[0]);
    //     p.b        = pixel[0];
    //     p.g        = pixel[1];
    //     p.r        = pixel[2];
    //   }
    // }
    // pcl::io::savePLYFileBinary(filename + ".ply", cloud_rgb);
  // }

  // colorize
  Global_map map;
  map.m_recent_visited_voxel_activated_time = 5;
  int img_id = 0;
  while (!scans.empty() && !camera_states.empty()) {
    if (!camera_states.front().exist) {
      img_id++;
      camera_states.pop_front();
      continue;
    }

    double lidar_time  = scans.front().header.stamp * 1e-6;
    double camera_time = camera_states.front().state.time;

    if (camera_time < lidar_time) {
      spdlog::info("{}", camera_states.size());
      spdlog::info("{:.10f} {:.10f}", camera_time, lidar_time);
      auto cam_state = camera_states.front().state;

      // img_ptr->m_gama_para.setZero();
      auto filename = (boost::format("/home/rick/Documents/raw_data/navvis/vlx/2022-07-28_03.43.39/colorize-kk/cam-export/%05d-cam0.jpg") % img_id).str();
      auto img      = cv::imread(filename);
      if (img.empty()) {
        camera_states.pop_front();
        continue;
      }

      std::shared_ptr<Image_frame> img_ptr = std::make_shared<Image_frame>();
      img_ptr->set_frame_idx(img_id);
      img_ptr->set_pose(cam_state.rot.conjugate(), -(cam_state.rot.conjugate() * cam_state.pos));
      img_ptr->m_cam_K << 910.6, 0, 2732,
          0, 910.6, 1820,
          0, 0, 1;
      img_ptr->set_intrinsic(img_ptr->m_cam_K);
      img_ptr->m_img        = img;
      map.render_with_a_image(img_ptr);

      img_id++;
      camera_states.pop_front();
    } else {
      auto &scan = scans.front();
      map.append_points_to_global_map(scan, scan.header.stamp * 1e-6);
      scans.pop_front();
    }
  }

  map.save_to_pcd("test.pcd", 1);

  return 0;
}
