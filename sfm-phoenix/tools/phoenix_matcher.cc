#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/matchers/lightglue.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/estimators/two_view_geometry.h>
#include <colmap/scene/database.h>

#include <migration/logging.h>

#include <algorithm>
#include <filesystem>
#include <fstream>
#include <spdlog/spdlog.h>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace {

struct CachedFeatures {
  colmap::FeatureKeypoints colmap_keypoints;
  std::vector<float> keypoints;
  std::vector<float> descriptors;
  int num_keypoints = 0;
};

std::vector<Eigen::Vector2d> ToPoints(
    const colmap::FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(keypoints.size());
  for (const auto& keypoint : keypoints) {
    points.emplace_back(keypoint.x, keypoint.y);
  }
  return points;
}

std::vector<std::string> ReadTextLines(const std::filesystem::path& path) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("Cannot open file: " + path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    if (!line.empty() && line[0] != '#') {
      lines.push_back(line);
    }
  }
  return lines;
}

std::vector<colmap::Image> LoadImages(const colmap::Database& database) {
  std::vector<colmap::Image> images = database.ReadAllImages();
  std::sort(images.begin(),
            images.end(),
            [](const colmap::Image& lhs, const colmap::Image& rhs) {
              return lhs.Name() < rhs.Name();
            });
  return images;
}

std::vector<std::pair<colmap::image_t, colmap::image_t>> BuildExhaustivePairs(
    const std::vector<colmap::Image>& images) {
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  for (size_t index0 = 0; index0 < images.size(); ++index0) {
    for (size_t index1 = index0 + 1; index1 < images.size(); ++index1) {
      pairs.emplace_back(images[index0].ImageId(), images[index1].ImageId());
    }
  }
  return pairs;
}

std::vector<std::pair<colmap::image_t, colmap::image_t>> ReadPairs(
    const std::filesystem::path& pair_list_path,
    const std::unordered_map<std::string, colmap::image_t>& image_ids) {
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  for (const auto& line : ReadTextLines(pair_list_path)) {
    std::istringstream stream(line);
    std::string image_name1;
    std::string image_name2;
    stream >> image_name1 >> image_name2;
    if (image_name1.empty() || image_name2.empty()) {
      continue;
    }

    const auto it1 = image_ids.find(image_name1);
    const auto it2 = image_ids.find(image_name2);
    if (it1 == image_ids.end() || it2 == image_ids.end()) {
      throw std::runtime_error("Pair list references unknown image: " + line);
    }
    if (it1->second == it2->second) {
      continue;
    }
    pairs.emplace_back(it1->second, it2->second);
  }
  return pairs;
}

const CachedFeatures& LoadFeatures(
    const colmap::Database& database,
    const std::unordered_map<colmap::image_t, colmap::Image>& images,
    const std::unordered_map<colmap::camera_t, colmap::Camera>& cameras,
    const int max_edge,
    const colmap::image_t image_id,
    std::unordered_map<colmap::image_t, CachedFeatures>* cache) {
  const auto cached = cache->find(image_id);
  if (cached != cache->end()) {
    return cached->second;
  }

  const auto image_it = images.find(image_id);
  if (image_it == images.end()) {
    throw std::runtime_error("Image missing from image cache");
  }

  const auto camera_it = cameras.find(image_it->second.CameraId());
  if (camera_it == cameras.end()) {
    throw std::runtime_error("Camera missing from camera cache");
  }

  const auto keypoints = database.ReadKeypoints(image_id);
  const auto descriptors = database.ReadDescriptors(image_id);
  if (static_cast<int>(keypoints.size()) != descriptors.rows()) {
    throw std::runtime_error("Keypoint / descriptor count mismatch for image: " +
                             image_it->second.Name());
  }
  if (sfm_phoenix::DescriptorDim(descriptors) != 128) {
    throw std::runtime_error(
        "Image does not contain Phoenix 128D float descriptors: " +
        image_it->second.Name());
  }

  CachedFeatures loaded;
  loaded.num_keypoints = static_cast<int>(keypoints.size());
  loaded.colmap_keypoints = keypoints;
  const float scale =
      sfm_phoenix::ComputeExtractionScale(camera_it->second, max_edge);
  loaded.keypoints = sfm_phoenix::ToMatcherKeypoints(keypoints, scale);
  loaded.descriptors = sfm_phoenix::ToMatcherDescriptors(descriptors);

  const auto [it, inserted] = cache->emplace(image_id, std::move(loaded));
  return it->second;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    InitSpdLog();

    constexpr int kMaxMatchedFeatures = 4000;
    std::filesystem::path pair_list_path;
    std::string lightglue_engine;
    int max_edge = 1600;
    int max_matches = kMaxMatchedFeatures;
    bool skip_existing = true;
    bool overwrite_existing = false;

    colmap::OptionManager options;
    options.AddDatabaseOptions();
    options.AddDefaultOption("pair_list_path", &pair_list_path);
    options.AddRequiredOption("Phoenix.lightglue_engine", &lightglue_engine);
    options.AddDefaultOption("Phoenix.max_edge", &max_edge);
    options.AddDefaultOption("Phoenix.max_matches", &max_matches);
    options.AddDefaultOption("Phoenix.skip_existing", &skip_existing);
    options.AddDefaultOption("Phoenix.overwrite_existing", &overwrite_existing);
    options.Parse(argc, argv);

    if (max_matches > kMaxMatchedFeatures) {
      spdlog::warn("Phoenix.max_matches={} exceeds hard cap {}, clamping",
                   max_matches,
                   kMaxMatchedFeatures);
      max_matches = kMaxMatchedFeatures;
    }

    auto database = colmap::Database::Open(*options.database_path);
    const std::vector<colmap::Image> all_images = LoadImages(*database);
    std::unordered_map<std::string, colmap::image_t> image_ids;
    std::unordered_map<colmap::image_t, colmap::Image> images_by_id;
    image_ids.reserve(all_images.size());
    images_by_id.reserve(all_images.size());
    for (const auto& image : all_images) {
      image_ids.emplace(image.Name(), image.ImageId());
      images_by_id.emplace(image.ImageId(), image);
    }

    std::unordered_map<colmap::camera_t, colmap::Camera> cameras_by_id;
    for (const auto& camera : database->ReadAllCameras()) {
      cameras_by_id.emplace(camera.camera_id, camera);
    }

    std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
    if (pair_list_path.empty()) {
      pairs = BuildExhaustivePairs(all_images);
    } else {
      pairs = ReadPairs(pair_list_path, image_ids);
    }

    sfm_phoenix::LightGlueConfig matcher_config;
    matcher_config.engine_path = lightglue_engine;
    matcher_config.max_matches = max_matches;

    sfm_phoenix::LightGlueMatcher matcher;
    if (!matcher.init(matcher_config)) {
      spdlog::error("Failed to initialize Phoenix LightGlue matcher");
      return EXIT_FAILURE;
    }

    std::unordered_map<colmap::image_t, CachedFeatures> feature_cache;
    int num_matched_pairs = 0;
    int num_skipped_pairs = 0;

    database->BeginTransaction();
    for (const auto& [image_id1, image_id2] : pairs) {
      const bool has_matches = database->ExistsMatches(image_id1, image_id2);
      const bool has_geometry =
          database->ExistsInlierMatches(image_id1, image_id2);
      if ((has_matches || has_geometry) && skip_existing && !overwrite_existing) {
        spdlog::info("Skip existing pair: {} <-> {}",
                     images_by_id.at(image_id1).Name(),
                     images_by_id.at(image_id2).Name());
        ++num_skipped_pairs;
        continue;
      }

      if (overwrite_existing) {
        if (has_geometry) {
          database->DeleteInlierMatches(image_id1, image_id2);
        }
        if (has_matches) {
          database->DeleteMatches(image_id1, image_id2);
        }
      } else if (has_matches || has_geometry) {
        ++num_skipped_pairs;
        continue;
      }

      const auto& features1 = LoadFeatures(*database,
                                           images_by_id,
                                           cameras_by_id,
                                           max_edge,
                                           image_id1,
                                           &feature_cache);
      const auto& features2 = LoadFeatures(*database,
                                           images_by_id,
                                           cameras_by_id,
                                           max_edge,
                                           image_id2,
                                           &feature_cache);
      if (features1.num_keypoints == 0 || features2.num_keypoints == 0) {
        ++num_matched_pairs;
        continue;
      }

      const auto matches = matcher.match(features1.keypoints.data(),
                                         features1.descriptors.data(),
                                         features1.num_keypoints,
                                         features2.keypoints.data(),
                                         features2.descriptors.data(),
                                         features2.num_keypoints);
      const auto colmap_matches = sfm_phoenix::ToColmapMatches(matches);
      if (matches.num_matches > 0) {
        database->WriteMatches(image_id1, image_id2, colmap_matches);

        const auto points1 = ToPoints(features1.colmap_keypoints);
        const auto points2 = ToPoints(features2.colmap_keypoints);
        colmap::TwoViewGeometryOptions geometry_options;
        colmap::TwoViewGeometry geometry = colmap::EstimateTwoViewGeometry(
            cameras_by_id.at(images_by_id.at(image_id1).CameraId()),
            points1,
            cameras_by_id.at(images_by_id.at(image_id2).CameraId()),
            points2,
            colmap_matches,
            geometry_options);
        database->WriteTwoViewGeometry(image_id1, image_id2, geometry);
      }
      ++num_matched_pairs;

      spdlog::info("Matched {} <-> {}: {}",
                   images_by_id.at(image_id1).Name(),
                   images_by_id.at(image_id2).Name(),
                   matches.num_matches);
    }
    database->EndTransaction();

    spdlog::info("Phoenix matching finished. matched_pairs={} skipped_pairs={}",
                 num_matched_pairs,
                 num_skipped_pairs);
    return EXIT_SUCCESS;
  } catch (const std::exception& error) {
    spdlog::error("{}", error.what());
    return EXIT_FAILURE;
  }
}