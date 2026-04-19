#include "feature_extraction/aliked_trt.h"
#include "feature_extraction/colmap_bridge.h"

#include <colmap/controllers/image_reader.h>
#include <colmap/controllers/option_manager.h>
#include <colmap/scene/database.h>

#include <opencv2/imgcodecs.hpp>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace {

void ApplyCameraMode(const int camera_mode,
                     colmap::ImageReaderOptions* reader_options) {
  switch (camera_mode) {
    case 0:
      reader_options->single_camera = false;
      reader_options->single_camera_per_folder = false;
      reader_options->single_camera_per_image = false;
      break;
    case 1:
      reader_options->single_camera = true;
      reader_options->single_camera_per_folder = false;
      reader_options->single_camera_per_image = false;
      break;
    case 2:
      reader_options->single_camera = false;
      reader_options->single_camera_per_folder = true;
      reader_options->single_camera_per_image = false;
      break;
    case 3:
      reader_options->single_camera = false;
      reader_options->single_camera_per_folder = false;
      reader_options->single_camera_per_image = true;
      break;
    default:
      throw std::runtime_error("Unsupported camera_mode: " +
                               std::to_string(camera_mode));
  }
}

std::vector<std::string> ReadTextLines(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("Cannot open file: " + path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty()) {
      lines.push_back(line);
    }
  }
  return lines;
}

void ImportImages(const colmap::ImageReaderOptions& reader_options,
                  colmap::Database* database) {
  colmap::ImageReader image_reader(reader_options, database);
  while (image_reader.NextIndex() < image_reader.NumImages()) {
    colmap::Rig rig;
    colmap::Camera camera;
    colmap::Image image;
    colmap::PosePrior pose_prior;
    colmap::Bitmap bitmap;
    const auto status =
        image_reader.Next(&rig, &camera, &image, &pose_prior, &bitmap, nullptr);
    if (status == colmap::ImageReader::Status::SUCCESS ||
        status == colmap::ImageReader::Status::IMAGE_EXISTS) {
      continue;
    }

    throw std::runtime_error("ImageReader failed on index " +
                             std::to_string(image_reader.NextIndex()) + ": " +
                             colmap::ImageReader::StatusToString(status));
  }
}

std::vector<colmap::Image> CollectTargetImages(
    const colmap::Database& database,
    const std::vector<std::string>& image_names) {
  std::vector<colmap::Image> all_images = database.ReadAllImages();
  std::sort(all_images.begin(),
            all_images.end(),
            [](const colmap::Image& lhs, const colmap::Image& rhs) {
              return lhs.Name() < rhs.Name();
            });

  if (image_names.empty()) {
    return all_images;
  }

  std::unordered_map<std::string, colmap::Image> images_by_name;
  images_by_name.reserve(all_images.size());
  for (const auto& image : all_images) {
    images_by_name.emplace(image.Name(), image);
  }

  std::vector<colmap::Image> selected_images;
  selected_images.reserve(image_names.size());
  for (const auto& image_name : image_names) {
    const auto it = images_by_name.find(image_name);
    if (it == images_by_name.end()) {
      throw std::runtime_error("Image not found in database: " + image_name);
    }
    selected_images.push_back(it->second);
  }
  return selected_images;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    std::filesystem::path image_list_path;
    int camera_mode = -1;
    std::string backbone_engine;
    std::string sddh_engine;
    int max_edge = 1600;
    int top_k = 5000;
    double scores_th = 0.2;
    bool skip_existing = true;

    colmap::OptionManager options;
    options.AddDatabaseOptions();
    options.AddImageOptions();
    options.AddDefaultOption("camera_mode", &camera_mode);
    options.AddDefaultOption("image_list_path", &image_list_path);
    options.AddFeatureExtractionOptions();
    options.AddRequiredOption("Phoenix.backbone_engine", &backbone_engine);
    options.AddRequiredOption("Phoenix.sddh_engine", &sddh_engine);
    options.AddDefaultOption("Phoenix.max_edge", &max_edge);
    options.AddDefaultOption("Phoenix.top_k", &top_k);
    options.AddDefaultOption("Phoenix.scores_th", &scores_th);
    options.AddDefaultOption("Phoenix.skip_existing", &skip_existing);
    options.Parse(argc, argv);

    colmap::ImageReaderOptions reader_options = *options.image_reader;
    reader_options.image_path = *options.image_path;
    reader_options.as_rgb = true;

    if (camera_mode >= 0) {
      ApplyCameraMode(camera_mode, &reader_options);
    }

    if (!image_list_path.empty()) {
      reader_options.image_names = ReadTextLines(image_list_path);
      if (reader_options.image_names.empty()) {
        return EXIT_SUCCESS;
      }
    }

    reader_options.Check();

    auto database = colmap::Database::Open(*options.database_path);
    ImportImages(reader_options, database.get());

    const std::vector<colmap::Image> images =
        CollectTargetImages(*database, reader_options.image_names);

    feature_extraction::AlikedConfig detector_config;
    detector_config.backbone_engine = backbone_engine;
    detector_config.sddh_engine = sddh_engine;
    detector_config.max_edge = max_edge;
    detector_config.dkd.top_k = top_k;
    detector_config.dkd.scores_th = static_cast<float>(scores_th);

    feature_extraction::AlikedDetector detector;
    if (!detector.init(detector_config)) {
      std::cerr << "Failed to initialize Phoenix ALIKED detector" << std::endl;
      return EXIT_FAILURE;
    }

    int num_extracted = 0;
    int num_skipped = 0;
    database->BeginTransaction();
    for (const auto& image : images) {
      const auto image_id = image.ImageId();
      const bool has_keypoints = database->ExistsKeypoints(image_id);
      const bool has_descriptors = database->ExistsDescriptors(image_id);
      if (has_keypoints || has_descriptors) {
        if (skip_existing) {
          ++num_skipped;
          continue;
        }
        std::cerr << "Image already has features in database: "
                  << image.Name() << std::endl;
        database->EndTransaction();
        return EXIT_FAILURE;
      }

        const std::filesystem::path image_path =
          std::filesystem::path(reader_options.image_path) / image.Name();
      const cv::Mat image_bgr =
          cv::imread(image_path.string(), cv::IMREAD_COLOR);
      if (image_bgr.empty()) {
        std::cerr << "Cannot read image: " << image_path << std::endl;
        database->EndTransaction();
        return EXIT_FAILURE;
      }

      const auto result = detector.detect(image_bgr);
      const auto keypoints = feature_extraction::ToColmapKeypoints(result);
      const auto descriptors = feature_extraction::ToColmapDescriptors(result);

      database->WriteKeypoints(image_id, keypoints);
      database->WriteDescriptors(image_id, descriptors);
      ++num_extracted;

      std::cout << "Extracted " << result.num_keypoints << " features for "
                << image.Name() << std::endl;
    }
    database->EndTransaction();

    std::cout << "Phoenix feature extraction finished. extracted="
              << num_extracted << " skipped=" << num_skipped << std::endl;
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}