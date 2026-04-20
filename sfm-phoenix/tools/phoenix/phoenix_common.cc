#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"

#include <colmap/estimators/two_view_geometry.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <cctype>
#include <chrono>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace phoenix_tool {

namespace {

std::string QuoteArg(const std::string& argument) {
  std::string quoted = "\"";
  for (const char ch : argument) {
    if (ch == '\"') {
      quoted += '\\';
    }
    quoted += ch;
  }
  quoted += '"';
  return quoted;
}

std::string ToLower(std::string value) {
  std::transform(value.begin(),
                 value.end(),
                 value.begin(),
                 [](const unsigned char ch) {
                   return static_cast<char>(std::tolower(ch));
                 });
  return value;
}

bool IsSupportedImageExtension(const std::filesystem::path& path) {
  static const std::unordered_set<std::string> kExtensions = {
      ".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"};
  return kExtensions.find(ToLower(path.extension().string())) !=
         kExtensions.end();
}

cv::Mat LoadStaticFilterFrame(const std::filesystem::path& image_path) {
  const cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
  if (image_bgr.empty()) {
    throw std::runtime_error("Cannot read image: " + image_path.string());
  }

  cv::Mat image_gray;
  cv::cvtColor(image_bgr, image_gray, cv::COLOR_BGR2GRAY);
  cv::resize(image_gray,
             image_gray,
             cv::Size(256, 256),
             0.0,
             0.0,
             cv::INTER_AREA);
  return image_gray;
}

double ComputeAdjacentFrameDifference(const cv::Mat& previous_frame,
                                      const cv::Mat& current_frame) {
  cv::Mat diff;
  cv::absdiff(previous_frame, current_frame, diff);
  return cv::mean(diff)[0];
}

uint64_t PairKey(const colmap::image_t image_id1,
                 const colmap::image_t image_id2) {
  const auto min_id = static_cast<uint32_t>(std::min(image_id1, image_id2));
  const auto max_id = static_cast<uint32_t>(std::max(image_id1, image_id2));
  return (static_cast<uint64_t>(min_id) << 32) | max_id;
}

}  // namespace

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

std::vector<Eigen::Vector2d> ToPoints(
    const colmap::FeatureKeypoints& keypoints) {
  std::vector<Eigen::Vector2d> points;
  points.reserve(keypoints.size());
  for (const auto& keypoint : keypoints) {
    points.emplace_back(keypoint.x, keypoint.y);
  }
  return points;
}

int RunCommand(const std::string& executable,
               const std::vector<std::string>& args,
               const std::string& prefix) {
#ifdef _WIN32
  std::ostringstream command_line;
  command_line << QuoteArg(executable);
  if (!prefix.empty()) {
    command_line << ' ' << QuoteArg(prefix);
  }
  for (const auto& arg : args) {
    command_line << ' ' << QuoteArg(arg);
  }

  const std::string narrow_command = command_line.str();

  STARTUPINFOW startup_info;
  PROCESS_INFORMATION process_info;
  ZeroMemory(&startup_info, sizeof(startup_info));
  ZeroMemory(&process_info, sizeof(process_info));
  startup_info.cb = sizeof(startup_info);

  std::wstring wide_command(narrow_command.begin(), narrow_command.end());
  std::vector<wchar_t> mutable_command(wide_command.begin(),
                                       wide_command.end());
  mutable_command.push_back(L'\0');

  if (!CreateProcessW(nullptr,
                      mutable_command.data(),
                      nullptr,
                      nullptr,
                      TRUE,
                      0,
                      nullptr,
                      nullptr,
                      &startup_info,
                      &process_info)) {
    throw std::runtime_error("Failed to launch subcommand: " + executable);
  }

  WaitForSingleObject(process_info.hProcess, INFINITE);
  DWORD exit_code = 1;
  GetExitCodeProcess(process_info.hProcess, &exit_code);
  CloseHandle(process_info.hThread);
  CloseHandle(process_info.hProcess);
  return static_cast<int>(exit_code);
#else
  std::ostringstream command;
  command << QuoteArg(executable);
  if (!prefix.empty()) {
    command << ' ' << QuoteArg(prefix);
  }
  for (const auto& arg : args) {
    command << ' ' << QuoteArg(arg);
  }
  return std::system(command.str().c_str());
#endif
}

std::filesystem::path ExecutableDirectory(const char* argv0) {
  return std::filesystem::absolute(argv0).parent_path();
}

void PrintHelp() {
  std::cout
      << "Phoenix commands:\n"
      << "  feature_extractor        Extract Phoenix features into a "
         "COLMAP database\n"
      << "  feature_matcher          Match Phoenix features in a COLMAP "
         "database\n"
      << "  reconstruction           Run COLMAP mapper for sparse "
         "reconstruction\n"
      << "  automatic_reconstructor  Extract + match + reconstruct in "
         "one command\n";
}

std::vector<std::string> SliceArgs(int argc, char** argv, int start) {
  std::vector<std::string> args;
  for (int index = start; index < argc; ++index) {
    args.emplace_back(argv[index]);
  }
  return args;
}

std::vector<std::string> ReadTextLines(const std::filesystem::path& path,
                                       const bool allow_comments) {
  std::ifstream stream(path);
  if (!stream.is_open()) {
    throw std::runtime_error("Cannot open file: " + path.string());
  }

  std::vector<std::string> lines;
  std::string line;
  while (std::getline(stream, line)) {
    if (line.empty()) {
      continue;
    }
    if (allow_comments && line[0] == '#') {
      continue;
    }
    lines.push_back(line);
  }
  return lines;
}

std::vector<std::string> CollectImageNamesFromDirectory(
    const std::filesystem::path& image_path) {
  std::vector<std::string> image_names;
  for (const auto& entry : std::filesystem::directory_iterator(image_path)) {
    if (!entry.is_regular_file() || !IsSupportedImageExtension(entry.path())) {
      continue;
    }
    image_names.push_back(entry.path().filename().string());
  }
  std::sort(image_names.begin(), image_names.end());
  return image_names;
}

std::vector<std::string> FilterStaticAdjacentFrames(
    const std::filesystem::path& image_path,
    const std::vector<std::string>& image_names,
    const double static_frame_diff_threshold,
    int* num_filtered_static_frames) {
  if (image_names.empty()) {
    return image_names;
  }

  std::vector<std::string> filtered_image_names;
  filtered_image_names.reserve(image_names.size());
  filtered_image_names.push_back(image_names.front());

  cv::Mat previous_kept_frame =
      LoadStaticFilterFrame(image_path / image_names.front());
  *num_filtered_static_frames = 0;

  for (size_t index = 1; index < image_names.size(); ++index) {
    const cv::Mat current_frame =
        LoadStaticFilterFrame(image_path / image_names[index]);
    const double frame_difference =
        ComputeAdjacentFrameDifference(previous_kept_frame, current_frame);
    if (frame_difference <= static_frame_diff_threshold) {
      ++(*num_filtered_static_frames);
      spdlog::info(
          "Filter static adjacent frame: {} diff_mean={:.3f} "
          "threshold={:.3f}",
          image_names[index],
          frame_difference,
          static_frame_diff_threshold);
      continue;
    }

    filtered_image_names.push_back(image_names[index]);
    previous_kept_frame = current_frame;
  }

  return filtered_image_names;
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
    if (status == colmap::ImageReader::Status::SUCCESS) {
      if (image.ImageId() == colmap::kInvalidImageId) {
        const auto image_id = database->WriteImage(image);
        image.SetImageId(image_id);

        if (pose_prior.IsValid()) {
          database->WritePosePrior(image_id, pose_prior);
        }

        colmap::Frame frame;
        frame.SetRigId(rig.RigId());
        frame.AddDataId(image.DataId());
        database->WriteFrame(frame);
      }
      continue;
    }

    if (status == colmap::ImageReader::Status::IMAGE_EXISTS) {
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

std::vector<colmap::Image> LoadImages(const colmap::Database& database) {
  std::vector<colmap::Image> images = database.ReadAllImages();
  std::sort(images.begin(),
            images.end(),
            [](const colmap::Image& lhs, const colmap::Image& rhs) {
              return lhs.Name() < rhs.Name();
            });
  return images;
}

std::vector<colmap::Image> SelectOrderedImages(
    const std::vector<colmap::Image>& all_images,
    const std::vector<std::string>& image_names) {
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
      spdlog::warn("Skip image missing from database: {}", image_name);
      continue;
    }
    selected_images.push_back(it->second);
  }
  return selected_images;
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

std::vector<std::pair<colmap::image_t, colmap::image_t>> BuildSequentialPairs(
    const std::vector<colmap::Image>& images,
    const int linear_overlap_num,
    const int quadratic_overlap_num) {
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  std::unordered_set<uint64_t> seen_pairs;

  auto try_add_pair = [&](const size_t index0, const size_t index1) {
    if (index1 >= images.size()) {
      return;
    }
    const auto image_id1 = images[index0].ImageId();
    const auto image_id2 = images[index1].ImageId();
    if (image_id1 == image_id2) {
      return;
    }
    const auto [_, inserted] = seen_pairs.insert(PairKey(image_id1, image_id2));
    if (inserted) {
      pairs.emplace_back(image_id1, image_id2);
    }
  };

  for (size_t index = 0; index < images.size(); ++index) {
    for (int offset = 1; offset <= linear_overlap_num; ++offset) {
      try_add_pair(index, index + static_cast<size_t>(offset));
    }
    for (int exponent = 1; exponent <= quadratic_overlap_num; ++exponent) {
      try_add_pair(index, index + (static_cast<size_t>(1) << exponent));
    }
  }

  return pairs;
}

std::vector<std::pair<colmap::image_t, colmap::image_t>> ReadPairs(
    const std::filesystem::path& pair_list_path,
    const std::unordered_map<std::string, colmap::image_t>& image_ids) {
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  for (const auto& line : ReadTextLines(pair_list_path, true)) {
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
    std::unordered_map<colmap::image_t, CachedFeatures>* cache,
    FeatureMatchingMetrics* metrics) {
  using Clock = std::chrono::steady_clock;
  const auto load_start = Clock::now();

  const auto cached = cache->find(image_id);
  if (cached != cache->end()) {
    if (metrics != nullptr) {
      metrics->feature_cache_hits += 1;
      metrics->feature_load_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() - load_start)
              .count());
    }
    return cached->second;
  }

  if (metrics != nullptr) {
    metrics->feature_cache_misses += 1;
  }

  const auto image_it = images.find(image_id);
  if (image_it == images.end()) {
    throw std::runtime_error("Image missing from image cache");
  }

  const auto camera_it = cameras.find(image_it->second.CameraId());
  if (camera_it == cameras.end()) {
    throw std::runtime_error("Camera missing from camera cache");
  }

  const auto db_read_start = Clock::now();
  const auto keypoints = database.ReadKeypoints(image_id);
  const auto descriptors = database.ReadDescriptors(image_id);
  if (metrics != nullptr) {
    metrics->feature_db_read_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - db_read_start)
            .count());
  }

  if (static_cast<int>(keypoints.size()) != descriptors.rows()) {
    throw std::runtime_error("Keypoint / descriptor count mismatch for image: " +
                             image_it->second.Name());
  }
  if (sfm_phoenix::DescriptorDim(descriptors) != 128) {
    throw std::runtime_error(
        "Image does not contain Phoenix 128D float descriptors: " +
        image_it->second.Name());
  }

  const auto convert_start = Clock::now();
  CachedFeatures loaded;
  loaded.num_keypoints = static_cast<int>(keypoints.size());
  loaded.colmap_keypoints = keypoints;
  loaded.points = ToPoints(keypoints);
  const float scale =
      sfm_phoenix::ComputeExtractionScale(camera_it->second, max_edge);
  loaded.keypoints = sfm_phoenix::ToMatcherKeypoints(keypoints, scale);
  loaded.descriptors = sfm_phoenix::ToMatcherDescriptors(descriptors);
  if (metrics != nullptr) {
    metrics->feature_convert_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - convert_start)
            .count());
  }

  const auto [it, inserted] = cache->emplace(image_id, std::move(loaded));
  if (metrics != nullptr) {
    metrics->feature_load_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - load_start)
            .count());
    metrics->cached_images = static_cast<int>(cache->size());
  }
  return it->second;
}

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "True";
}

}  // namespace phoenix_tool