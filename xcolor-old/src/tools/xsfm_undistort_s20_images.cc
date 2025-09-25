#include <glog/logging.h>
#include <sqlite3.h>
#include <yaml-cpp/yaml.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <cstring>
#include <iostream>
#include <opencv2/opencv.hpp>

#include "internal/poly_fisheye_camera.h"

int kOutputImageWidth  = 6000;
int kOutputImageHeight = 8000;

DEFINE_string(project_dir, "D:/ProjectX/project-3d/data/sfm/2025-05-08_15-02-53/output/", "Project directory path");
DEFINE_string(calibration_filename, "D:/ProjectX/project-3d/data/sfm/2025-05-08_15-02-53/output/calibration.yaml", "Camera calibration file path");

namespace fs = boost::filesystem;

// Calculate angle between two vectors in radians
double CalculateAngle(const Eigen::Vector3d& v1, const Eigen::Vector3d& v2) {
  double dot_product = v1.dot(v2) / (v1.norm() * v2.norm());
  // Clamp to valid range for acos
  dot_product = std::max(std::min(dot_product, 1.0), -1.0);
  return std::acos(dot_product);
}

// Generate angle difference map for all pixels
cv::Mat GenerateAngleDifferenceMap(const camera_model::PolyFisheyeCamera& camera, int width, int height, const std::string& camera_type) {
  LOG(INFO) << "Generating angle difference map for " << camera_type << " camera...";

  // Create output map: 32-bit float for accurate angle values
  cv::Mat angle_map(height, width, CV_32FC1, cv::Scalar(0));

  // Define the max step size for processing (to reduce computation time)
  int count = 0;

#pragma omp parallel for
  for (int y = 0; y < height; y += 1) {
#pragma omp critical
    {
      LOG(INFO) << ++count;
    }
    for (int x = 0; x < width; x += 1) {
      // Get 3D vectors for current pixel and diagonal neighbor
      Eigen::Vector3d v1, v2;

      // Current pixel (x,y)
      camera.liftSphere(Eigen::Vector2d(x, y), v1);

      // Diagonal neighbor pixel (x+1,y+1)
      camera.liftSphere(Eigen::Vector2d(x + 1, y + 1), v2);

      // Calculate angle between vectors (in radians)
      double angle_diff = 180 / M_PI * CalculateAngle(v1, v2);

      // Store in the map
      angle_map.at<float>(y, x) = angle_diff;
    }
  }

  return angle_map;
}

// Save angle difference map to file
void SaveAngleDifferenceMap(const cv::Mat& angle_map, const fs::path& output_dir, const std::string& camera_type) {
  fs::path output_file = output_dir / (camera_type + "_angle_diff_map.tiff");
  cv::imwrite(output_file.string(), angle_map);
  LOG(INFO) << "Angle difference map saved to: " << output_file.string();
}

// Process all images for a single camera
void UndistortAndSaveImages(const std::string& camera_type, const fs::path& input_dir, const fs::path& output_dir,
                   const camera_model::PolyFisheyeCamera& camera) {
  // Create corresponding output directory
  fs::path camera_output_dir = output_dir / camera_type;
  if (!fs::exists(camera_output_dir)) {
    fs::create_directories(camera_output_dir);
  }

  cv::Mat map1, map2;
  camera.initUndistortRectifyMap(map1, map2, camera.getParameters().A11(), camera.getParameters().A22(),
                                 cv::Size(kOutputImageWidth, kOutputImageHeight), kOutputImageWidth / 2, kOutputImageHeight / 2);
  DLOG(INFO) << "Undistort map initialized for " << camera_type << " camera.";

  // Traverse all images in the camera directory
  for (const auto& entry : fs::directory_iterator(input_dir)) {
    if (fs::is_regular_file(entry) &&
        (entry.path().extension() == ".jpg" || entry.path().extension() == ".png" || entry.path().extension() == ".jpeg")) {
      // Read image
      std::string image_path = entry.path().string();
      cv::Mat img            = cv::imread(image_path);

      if (img.empty()) {
        LOG(WARNING) << "Failed to read image: " << image_path;
        continue;
      }

      // Undistort processing
      cv::Mat undistorted;
      cv::remap(img, undistorted, map1, map2, cv::INTER_CUBIC, cv::BORDER_CONSTANT);

      // Save results to corresponding output directory
      fs::path output_file = camera_output_dir / entry.path().filename();
      cv::imwrite(output_file.string(), undistorted);

      DLOG(INFO) << "Processed " << entry.path().filename() << " (" << camera_type << ")";
    }
  }
}

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  fs::path project_dir(FLAGS_project_dir);
  fs::path images_dir    = project_dir / "images";
  fs::path undistort_dir = project_dir / "undistort";

  // Check if images directory exists
  if (!fs::exists(images_dir)) {
    LOG(ERROR) << "Images directory not found: " << images_dir;
    return -1;
  }

  // Create undistort output directory
  if (!fs::exists(undistort_dir)) {
    fs::create_directories(undistort_dir);
  }

  // Process left camera images
  fs::path left_dir = images_dir / "left";
  if (fs::exists(left_dir) && fs::is_directory(left_dir)) {
    camera_model::PolyFisheyeCamera::Parameters left_params;
    left_params.readFromYamlFile(FLAGS_calibration_filename, {"intrinsic", "fisheye_left"});
    camera_model::PolyFisheyeCamera left_camera{left_params};

    // Generate and save angle difference map for left camera
    // cv::Mat left_angle_map = GenerateAngleDifferenceMap(left_camera, left_camera.imageWidth(), left_camera.imageHeight(), "left");
    // SaveAngleDifferenceMap(left_angle_map, undistort_dir, "left");
    // exit(1);

    UndistortAndSaveImages("left", left_dir, undistort_dir, left_camera);
    LOG(INFO) << "Processed all left camera images.";
  } else {
    LOG(WARNING) << "Left camera directory not found: " << left_dir;
  }

  // Process right camera images
  fs::path right_dir = images_dir / "right";
  if (fs::exists(right_dir) && fs::is_directory(right_dir)) {
    camera_model::PolyFisheyeCamera::Parameters right_params;
    right_params.readFromYamlFile(FLAGS_calibration_filename, {"intrinsic", "fisheye_right"});
    camera_model::PolyFisheyeCamera right_camera{right_params};

    UndistortAndSaveImages("right", right_dir, undistort_dir, right_camera);
    LOG(INFO) << "Processed all right camera images.";
  } else {
    LOG(WARNING) << "Right camera directory not found: " << right_dir;
  }

  LOG(INFO) << "All images processed successfully.";
  std::cout << "All images processed, results saved in: " << undistort_dir << std::endl;

  return 0;
}
