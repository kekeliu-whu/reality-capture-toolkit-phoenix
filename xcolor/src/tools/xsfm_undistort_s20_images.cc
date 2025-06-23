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

// Process all images for a single camera
void ProcessImages(const std::string& camera_type, const fs::path& input_dir, const fs::path& output_dir,
                   const camera_model::PolyFisheyeCamera& camera) {
  // Create corresponding output directory
  fs::path camera_output_dir = output_dir / camera_type;
  if (!fs::exists(camera_output_dir)) {
    fs::create_directories(camera_output_dir);
  }

  cv::Mat map1, map2;
  camera.initUndistortRectifyMap(map1, map2, camera.getParameters().A11(), camera.getParameters().A22(), cv::Size(kOutputImageWidth, kOutputImageHeight), kOutputImageWidth / 2,
                                 kOutputImageHeight / 2);
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

    ProcessImages("left", left_dir, undistort_dir, left_camera);
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

    ProcessImages("right", right_dir, undistort_dir, right_camera);
    LOG(INFO) << "Processed all right camera images.";
  } else {
    LOG(WARNING) << "Right camera directory not found: " << right_dir;
  }

  LOG(INFO) << "All images processed successfully.";
  std::cout << "All images processed, results saved in: " << undistort_dir << std::endl;

  return 0;
}
