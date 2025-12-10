#include <gflags/gflags.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <proj.h>
#include <spdlog/sinks/basic_file_sink.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <Eigen/Eigen>
#include <algorithm>
#include <boost/algorithm/string.hpp>
#include <boost/algorithm/string/split.hpp>
#include <boost/filesystem.hpp>
#include <boost/format.hpp>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <map>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/io/BufferReader.hpp>
#include <pdal/io/LasWriter.hpp>
#include <sstream>
#include <vector>

#include "migration/logging.h"
#include "migration/string.h"

DEFINE_string(calibration_file, "C:\\4.indoor-big-slow\\hall\\CameraCalibration.json",
              "Path to camera calibration JSON file");
DEFINE_string(trajectory_file, "C:\\4.indoor-big-slow\\hall\\out\\traj_aligned.txt", "Path to trajectory file");
DEFINE_string(timestamp_file, "C:\\4.indoor-big-slow\\hall\\GPS_Time_Result.txt", "Path to timestamp file");
DEFINE_string(cloud_path, "C:\\4.indoor-big-slow\\hall\\out\\map_smooth.las", "Path to input point cloud LAS file");
DEFINE_string(colored_cloud_path, "C:\\4.indoor-big-slow\\hall\\out\\map_smooth_colored.las",
              "Path to output colored point cloud LAS file");
DEFINE_string(images_path, "C:\\4.indoor-big-slow\\hall\\camera\\", "Path to input images folder");

struct CameraCalibration {
  double          fx, fy, cx, cy;  // Focal length and principal point
  double          k1, k2, k3, k4;  // Distortion parameters
  int             width, height;   // Image resolution
  Eigen::Matrix3d K;               // Intrinsic matrix
  std::string     camera_name;
  // Extrinsics (relative to lidar)
  Eigen::Vector3d    t_c2l;  // Camera position relative to lidar
  Eigen::Quaterniond R_c2l;  // Camera rotation relative to lidar
};

struct PoseStamped {
  double             timestamp;
  Eigen::Vector3d    position;
  Eigen::Quaterniond orientation;
};

struct Image {
  double             timestamp;
  std::string        filename;
  CameraCalibration  calib;
  Eigen::Quaterniond R_w2c;
  Eigen::Vector3d    t_w2c;
};

// Read camera calibration file in JSON format
bool readCameraCalibration(const std::string& filename, const std::string& cameraName, CameraCalibration& calib) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Failed to open calibration file: {}", filename);
    return false;
  }

  nlohmann::json root;
  try {
    file >> root;
  } catch (const std::exception& e) {
    spdlog::error("JSON parsing error: {}", e.what());
    return false;
  }

  // Access specific camera intrinsics
  if (!root.contains("CameraInfo")) {
    spdlog::error("CameraInfo field not found in calibration file");
    return false;
  }

  if (!root["CameraInfo"].contains(cameraName)) {
    spdlog::error("Camera not found in calibration file: {}", cameraName);
    return false;
  }

  const nlohmann::json& cameraInfo = root["CameraInfo"][cameraName];

  if (!cameraInfo.contains("IntrinsicParameters")) {
    spdlog::error("IntrinsicParameters not found");
    return false;
  }

  const nlohmann::json& intrinsic = cameraInfo["IntrinsicParameters"]["PARAMS"];

  calib.camera_name = cameraName;
  calib.fx          = intrinsic["fx"].get<double>();
  calib.fy          = intrinsic["fy"].get<double>();
  calib.cx          = intrinsic["cx"].get<double>();
  calib.cy          = intrinsic["cy"].get<double>();
  calib.width       = intrinsic["width"].get<int>();
  calib.height      = intrinsic["height"].get<int>();
  calib.k1          = intrinsic["k1"].get<double>();
  calib.k2          = intrinsic["k2"].get<double>();
  calib.k3          = intrinsic.contains("k3") ? intrinsic["k3"].get<double>() : 0.0;
  calib.k4          = intrinsic.contains("k4") ? intrinsic["k4"].get<double>() : 0.0;

  // Read extrinsic parameters (camera relative to lidar)
  const nlohmann::json& extrinsic = cameraInfo["ExtrinsicParameters"];
  calib.t_c2l[0]                  = extrinsic["offset"]["x"].get<double>();
  calib.t_c2l[1]                  = extrinsic["offset"]["y"].get<double>();
  calib.t_c2l[2]                  = extrinsic["offset"]["z"].get<double>();

  double             qx = extrinsic["Quaternion"]["qx"].get<double>();
  double             qy = extrinsic["Quaternion"]["qy"].get<double>();
  double             qz = extrinsic["Quaternion"]["qz"].get<double>();
  double             qw = extrinsic["Quaternion"]["qw"].get<double>();
  Eigen::Quaterniond q_cam2lidar(qw, qx, qy, qz);
  calib.R_c2l = q_cam2lidar;

  // Build intrinsic matrix
  calib.K       = Eigen::Matrix3d::Identity();
  calib.K(0, 0) = calib.fx;
  calib.K(1, 1) = calib.fy;
  calib.K(0, 2) = calib.cx;
  calib.K(1, 2) = calib.cy;

  spdlog::info(
      "Camera calibration info ({}): Focal length: fx={}, fy={}; Principal point: cx={}, cy={}; Resolution: {} x {}",
      cameraName, calib.fx, calib.fy, calib.cx, calib.cy, calib.width, calib.height);
  spdlog::info("Distortion: k1={}, k2={}, k3={}, k4={}", calib.k1, calib.k2, calib.k3, calib.k4);
  spdlog::info("Extrinsics (camera to lidar): Position: [{} {} {}]; Quaternion: [{}, {}, {}, {}]", calib.t_c2l[0],
               calib.t_c2l[1], calib.t_c2l[2], qw, qx, qy, qz);

  return true;
}

// Read trajectory file
bool readTrajectory(const std::string& filename, std::vector<PoseStamped>& poses) {
  spdlog::info("Reading trajectory from {}", filename);

  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Failed to open trajectory file: {}", filename);
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    std::istringstream iss(line);
    PoseStamped        pose;
    double             qx, qy, qz, qw;

    if (!(iss >> pose.timestamp >> pose.position[0] >> pose.position[1] >> pose.position[2] >> qx >> qy >> qz >> qw)) {
      continue;
    }

    pose.orientation = Eigen::Quaterniond(qw, qx, qy, qz);
    poses.push_back(pose);
  }

  spdlog::info("Loaded trajectory points: {}", poses.size());
  return true;
}

// Read timestamp file (format: camera_path/frame_number timestamp)
bool readTimeStamps(const std::string& filename, const std::string& cameraName,
                    std::map<std::string, double>& timestamps) {
  std::ifstream file(filename);
  if (!file.is_open()) {
    spdlog::error("Failed to open timestamp file: {}", filename);
    return false;
  }

  std::string line;
  while (std::getline(file, line)) {
    if (line.empty() || line[0] == '#') continue;

    // Parse format: "CAM_L/00001 1445254113.203429"
    std::istringstream iss(line);
    std::string        imagePath;
    double             timestamp;

    if (!(iss >> imagePath >> timestamp)) {
      continue;
    }

    // Check if it belongs to the specified camera
    if (imagePath.find(cameraName) == 0) {
      timestamps[imagePath] = timestamp;
    }
  }

  spdlog::info("Loaded timestamps for camera {}: {} frames", cameraName, timestamps.size());
  return true;
}

// Fisheye projection function (manual implementation)
bool fisheyeProject(const Eigen::Vector3d& point3d, const CameraCalibration& calib, cv::Point2f& pixel) {
  double x = point3d[0];
  double y = point3d[1];
  double z = point3d[2];

  // Points behind the camera
  if (z < 0.01) {
    return false;
  }

  // Step 1: Calculate normalized image coordinates and projection angle
  double u = x / z;  // Normalized x coordinate
  double v = y / z;  // Normalized y coordinate

  double r     = std::sqrt(u * u + v * v);  // Distance from principal point
  double theta = std::atan(r);              // Angle from optical axis

  // Step 2: Apply polynomial distortion model (OpenCV fisheye standard)
  double theta2 = theta * theta;
  double theta4 = theta2 * theta2;
  double theta6 = theta4 * theta2;
  double theta8 = theta4 * theta4;

  double theta_d = theta * (1.0 + calib.k1 * theta2 + calib.k2 * theta4 + calib.k3 * theta6 + calib.k4 * theta8);

  // Step 3: Distorted normalized coordinates
  double scale = (r > 1e-6) ? (theta_d / r) : 1.0;
  double u_d   = scale * u;
  double v_d   = scale * v;

  // Step 4: Convert to pixel coordinates using intrinsic parameters
  pixel.x = calib.fx * u_d + calib.cx;
  pixel.y = calib.fy * v_d + calib.cy;

  // Step 5: Check if within image bounds
  if (pixel.x >= 0 && pixel.x < calib.width && pixel.y >= 0 && pixel.y < calib.height) {
    return true;
  }

  return false;
}

// Get the two nearest trajectory points and interpolate
bool getInterpolatedPose(const std::vector<PoseStamped>& poses, double timestamp, PoseStamped& interpolatedPose) {
  // Find the two nearest timestamps
  int beforeIdx = -1, afterIdx = -1;

  for (int i = 0; i < poses.size(); ++i) {
    if (poses[i].timestamp <= timestamp) {
      beforeIdx = i;
    }
    if (poses[i].timestamp >= timestamp && afterIdx == -1) {
      afterIdx = i;
      break;
    }
  }

  // If no surrounding two points found
  if (beforeIdx == -1 || afterIdx == -1) {
    return false;
  }

  // If the two points are the same (equal timestamps or only one point)
  if (beforeIdx == afterIdx) {
    interpolatedPose = poses[beforeIdx];
    return true;
  }

  // Interpolate calculation
  const PoseStamped& poseBefore = poses[beforeIdx];
  const PoseStamped& poseAfter  = poses[afterIdx];

  double timeDiff = poseAfter.timestamp - poseBefore.timestamp;
  double alpha    = (timestamp - poseBefore.timestamp) / timeDiff;

  // Linear interpolation of position
  interpolatedPose.position = poseBefore.position + alpha * (poseAfter.position - poseBefore.position);

  // Spherical linear interpolation of quaternion
  interpolatedPose.orientation = poseBefore.orientation.slerp(alpha, poseAfter.orientation);
  interpolatedPose.timestamp   = timestamp;

  return true;
}

// Build image list with poses and calibration
std::vector<Image> buildImageList(const std::string& timestampFile, const CameraCalibration& calib,
                                  const std::vector<PoseStamped>&      poses,
                                  const std::map<std::string, double>& timestamps) {
  std::vector<Image> imageList;

  spdlog::info("========== Building Image List ==========");

  // Process each timestamp
  size_t frameIdx = 0;
  for (const auto& entry : timestamps) {
    const std::string& imagePath = entry.first;
    double             timestamp = entry.second;

    frameIdx++;
    Image img;
    img.timestamp = timestamp;
    img.calib     = calib;
    img.filename  = FLAGS_images_path + "\\" + imagePath + ".jpg";

    // Get interpolated pose for this timestamp
    PoseStamped pose_l2w;
    if (!getInterpolatedPose(poses, img.timestamp, pose_l2w)) {
      spdlog::warn("Cannot find pose for {} (timestamp: {:.6f})", imagePath, img.timestamp);
      continue;
    }

    // Compute world to camera transformation
    // world -> lidar -> camera
    Eigen::Matrix3d R_w2l = pose_l2w.orientation.conjugate().toRotationMatrix();
    Eigen::Vector3d t_w2l = -R_w2l * pose_l2w.position;

    Eigen::Matrix3d R_l2c = calib.R_c2l.conjugate().toRotationMatrix();
    Eigen::Vector3d t_l2c = -(R_l2c * calib.t_c2l);

    img.R_w2c = R_l2c * R_w2l;
    img.t_w2c = R_l2c * t_w2l + t_l2c;

    imageList.push_back(img);

    if (frameIdx % 10 == 0) {
      spdlog::info("Processed {} images...", frameIdx);
    }
  }

  spdlog::info("Total images processed: {}", imageList.size());
  spdlog::info("Image list building completed!");

  return imageList;
}

std::vector<Image> LoadAllImages(const std::string& calibrationFile, const std::string& timestampFile,
                                 const std::vector<PoseStamped>& poses) {
  spdlog::info("Loading all images from cameras...");

  // Combine image list from both cameras
  std::vector<Image> allImages;

  // Camera names to process
  std::vector<std::string> cameraNames = {"CAM_L", "CAM_R"};

  // Process each camera
  for (const auto& cameraName : cameraNames) {
    spdlog::info("========== Processing {} ==========", cameraName);

    CameraCalibration calib;
    if (!readCameraCalibration(calibrationFile, cameraName, calib)) {
      spdlog::error("Failed to read calibration file for {}", cameraName);
      exit(1);
    }

    // Read timestamps for this camera
    std::map<std::string, double> timestamps;
    if (!readTimeStamps(timestampFile, cameraName, timestamps)) {
      spdlog::error("Failed to read timestamp file for {}", cameraName);
      exit(1);
    }

    std::vector<Image> imageList = buildImageList(timestampFile, calib, poses, timestamps);
    allImages.insert(allImages.end(), imageList.begin(), imageList.end());
  }

  spdlog::info("Total images loaded from all cameras: {}", allImages.size());

  return allImages;
}

#pragma pack(push, 1)
struct LidarPoint {
  double  timestamp;
  float   x;
  float   y;
  float   z;
  uint8_t intensity;
  uint8_t valid;
  uint8_t r;
  uint8_t g;
  uint8_t b;
};
#pragma pack(pop)

std::vector<LidarPoint> ReadLidarPoints(const std::string& filename) {
  spdlog::info("Reading lidar point cloud from {}", filename);

  pdal::StageFactory factory;
  pdal::Stage*       reader = factory.createStage("readers.las");
  pdal::Options      opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  std::vector<LidarPoint> points;
  points.reserve(view->size());
  double last_timestamp = 0.0;
  for (size_t i = 0; i < view->size(); ++i) {
    LidarPoint p;
    p.x         = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    p.y         = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    p.z         = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.timestamp = view->getFieldAs<double>(pdal::Dimension::Id::GpsTime, i);
    p.intensity = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);
    p.valid     = false;

    if (p.timestamp < last_timestamp) {
      spdlog::warn("Timestamps are not in ascending order at point {}: current {:.6f}, last {:.6f}", i, p.timestamp,
                   last_timestamp);
      continue;
    }
    last_timestamp = p.timestamp;

    points.push_back(p);
  }

  spdlog::info("Loaded {} lidar points from {}", points.size(), filename);

  return points;
}

void setFusedColor(LidarPoint& pt, const std::vector<std::pair<double, cv::Vec3b>>& color_candidates) {
  int min_cand_num           = 3;
  int color_inlier_threshold = 60;

  if (color_candidates.empty()) {
    return;
  }

  std::vector<std::pair<double, cv::Vec3b>> sorted_candidates(color_candidates.begin(), color_candidates.end());
  std::sort(sorted_candidates.begin(), sorted_candidates.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });

  if (sorted_candidates.size() <= min_cand_num) {
    cv::Vec3i color = {0, 0, 0};
    for (int j = 0; j < sorted_candidates.size(); ++j) {
      color += sorted_candidates[j].second;
    }
    pt.b     = color[0] / (int)sorted_candidates.size();
    pt.g     = color[1] / (int)sorted_candidates.size();
    pt.r     = color[2] / (int)sorted_candidates.size();
    pt.valid = true;
    return;
  }

  cv::Vec3b best_color;
  for (int k = 0; k < 3; ++k) {
    std::vector<uint8_t> color_channel_candidates;
    color_channel_candidates.reserve(sorted_candidates.size());
    for (int j = 0; j < sorted_candidates.size(); ++j) {
      color_channel_candidates.push_back(sorted_candidates[j].second[k]);
    }

    int mid_idx = color_channel_candidates.size() / 2;
    std::nth_element(color_channel_candidates.begin(), color_channel_candidates.begin() + mid_idx,
                     color_channel_candidates.end());
    uint8_t mid_color = color_channel_candidates[mid_idx];

    int color_sum = 0;
    int color_num = 0;
    for (int j = 0; j < color_channel_candidates.size(); ++j) {
      if (std::abs((int)color_channel_candidates[j] - (int)mid_color) < color_inlier_threshold) {
        color_sum += color_channel_candidates[j];
        color_num += 1;
      }
    }

    best_color[k] = color_sum / color_num;
  }

  pt.b     = best_color[0];
  pt.g     = best_color[1];
  pt.r     = best_color[2];
  pt.valid = true;
}

void ColorizeWindowPoints(const std::vector<Image>& windowImages, double winStart, double winEnd,
                          std::vector<LidarPoint>& lidarPoints) {
  int                  coloredPointCount = 0;
  std::vector<cv::Mat> loadedImages(windowImages.size());
  for (size_t i = 0; i < windowImages.size(); ++i) {
    loadedImages[i] = cv::imread(windowImages[i].filename);
    if (loadedImages[i].empty()) {
      spdlog::error("Failed to load image: {}", windowImages[i].filename);
      exit(1);
    }
  }

  // Process each point
  auto it_start = std::lower_bound(lidarPoints.begin(), lidarPoints.end(), winStart,
                                   [](const LidarPoint& pt, double val) { return pt.timestamp < val; });
  auto it_end   = std::upper_bound(it_start, lidarPoints.end(), winEnd,
                                   [](double val, const LidarPoint& pt) { return val < pt.timestamp; });
  int  begin_i  = it_start - lidarPoints.begin();
  int  end_i    = it_end - lidarPoints.begin();

#pragma omp parallel for reduction(+ : coloredPointCount)
  for (int i = begin_i; i < end_i; ++i) {
    LidarPoint& pt = lidarPoints[i];

    if (pt.valid) {
      continue;  // Already colored
    }

    // Point in world frame
    Eigen::Vector3d worldPoint(pt.x, pt.y, pt.z);

    std::vector<std::pair<double, cv::Vec3b>> color_candidates;
    // Try to colorize with each image in the window
    for (int j = 0; j < windowImages.size(); ++j) {
      const auto& img = windowImages[j];
      // Transform to camera frame
      Eigen::Vector3d cameraPoint = img.R_w2c * worldPoint + img.t_w2c;

      // Skip points behind the camera
      if (cameraPoint[2] < 0.01) {
        continue;
      }

      // Project to pixel
      cv::Point2f pixel;
      if (!fisheyeProject(cameraPoint, img.calib, pixel)) {
        continue;
      }

      // Round to nearest integer
      int px = static_cast<int>(std::round(pixel.x));
      int py = static_cast<int>(std::round(pixel.y));

      // Check if pixel is within image bounds
      if (px >= 0 && px < img.calib.width && py >= 0 && py < img.calib.height) {
        // Load image
        cv::Mat image = loadedImages[j];
        if (image.empty()) {
          continue;
        }

        // Get color from image
        cv::Vec3b color = image.at<cv::Vec3b>(py, px);

        double dist = cameraPoint.norm();
        color_candidates.emplace_back(dist, color);
      }
    }

    setFusedColor(pt, color_candidates);
  }

  spdlog::info("  Colored points in this window: {}", end_i - begin_i);
}

void ColorizeAllPointCloud(const std::vector<Image>& allImages, std::vector<LidarPoint>& lidarPoints) {
  spdlog::info("Colorizing all point cloud...");

  auto allImages_copy = allImages;
  std::sort(allImages_copy.begin(), allImages_copy.end(),
            [](const Image& a, const Image& b) { return a.timestamp < b.timestamp; });

  double SLD_WIN_PERIOD  = 1.0;  // seconds
  double SLD_WIN_PADDING = 1.0;  // seconds
  for (double t = allImages_copy.front().timestamp + SLD_WIN_PADDING;
       t + SLD_WIN_PERIOD <= allImages_copy.back().timestamp - SLD_WIN_PADDING; t += SLD_WIN_PERIOD) {
    double winStart = t;
    double winEnd   = t + SLD_WIN_PERIOD;
    spdlog::info("Colorizing points in time window: [{:.6f}, {:.6f}]", winStart, winEnd);
    // Collect images in this time window
    std::vector<Image> windowImages;
    for (const auto& img : allImages_copy) {
      if (img.timestamp >= winStart - SLD_WIN_PADDING && img.timestamp <= winEnd + SLD_WIN_PADDING) {
        windowImages.push_back(img);
      }
    }
    spdlog::info("  Total images in this window: {}", windowImages.size());
    // Colorize points in this time window
    ColorizeWindowPoints(windowImages, winStart, winEnd, lidarPoints);
  }
}

void SaveColoredPointCloud(const std::string& filename, const std::vector<LidarPoint>& lidarPoints) {
  spdlog::info("Saving colored point cloud to {}", filename);

  pdal::PointTable     table;
  pdal::PointLayoutPtr layout = table.layout();

  layout->registerDim(pdal::Dimension::Id::X);
  layout->registerDim(pdal::Dimension::Id::Y);
  layout->registerDim(pdal::Dimension::Id::Z);
  layout->registerDim(pdal::Dimension::Id::Intensity);
  layout->registerDim(pdal::Dimension::Id::GpsTime);
  layout->registerDim(pdal::Dimension::Id::Red);
  layout->registerDim(pdal::Dimension::Id::Green);
  layout->registerDim(pdal::Dimension::Id::Blue);

  pdal::PointViewPtr view(new pdal::PointView(table));

  for (const auto& pt : lidarPoints) {
    if (!pt.valid) {
      continue;  // Skip uncolored points
    }

    pdal::PointId idx = view->size();
    view->setField(pdal::Dimension::Id::X, idx, pt.x);
    view->setField(pdal::Dimension::Id::Y, idx, pt.y);
    view->setField(pdal::Dimension::Id::Z, idx, pt.z);
    view->setField(pdal::Dimension::Id::Intensity, idx, static_cast<float>(pt.intensity));
    view->setField(pdal::Dimension::Id::GpsTime, idx, pt.timestamp);
    view->setField(pdal::Dimension::Id::Red, idx, pt.r);
    view->setField(pdal::Dimension::Id::Green, idx, pt.g);
    view->setField(pdal::Dimension::Id::Blue, idx, pt.b);
  }

  pdal::BufferReader reader;
  reader.addView(view);

  pdal::Options options;
  options.add("filename", PlatformToUTF8(filename));
  options.add("scale_x", 1e-4);
  options.add("scale_y", 1e-4);
  options.add("scale_z", 1e-4);
  options.add("offset_x", "auto");
  options.add("offset_y", "auto");
  options.add("offset_z", "auto");

  pdal::LasWriter writer;
  writer.setOptions(options);
  writer.setInput(reader);
  writer.prepare(table);
  writer.execute(table);

  spdlog::info("Saved colored point cloud to {}", filename);
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  InitSpdLog();

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);
  spdlog::info("Using {} / {} cores.", cores_used, cores);

  // File paths
  std::string calibrationFile  = FLAGS_calibration_file;
  std::string trajectoryFile   = FLAGS_trajectory_file;
  std::string timestampFile    = FLAGS_timestamp_file;
  std::string cloudPath        = FLAGS_cloud_path;
  std::string coloredCloudPath = FLAGS_colored_cloud_path;

  // Read trajectory
  std::vector<PoseStamped> poses;
  if (!readTrajectory(trajectoryFile, poses)) {
    spdlog::error("Failed to read trajectory file");
    return 1;
  }

  std::vector<Image> allImages = LoadAllImages(calibrationFile, timestampFile, poses);

  std::vector<LidarPoint> lidarPoints = ReadLidarPoints(cloudPath);

  ColorizeAllPointCloud(allImages, lidarPoints);

  SaveColoredPointCloud(coloredCloudPath, lidarPoints);

  return 0;
}
