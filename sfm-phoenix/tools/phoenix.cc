#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/model_runtime.h"
#include "sfm_phoenix/matchers/lightglue.h"

#include <colmap/controllers/image_reader.h>
#include <colmap/controllers/option_manager.h>
#include <colmap/estimators/two_view_geometry.h>
#include <colmap/scene/database.h>

#include <migration/logging.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cctype>
#include <chrono>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <cstdint>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#ifdef _WIN32
#include <windows.h>
#endif

namespace {

struct CachedFeatures {
  colmap::FeatureKeypoints colmap_keypoints;
  std::vector<float> keypoints;
  std::vector<float> descriptors;
  int num_keypoints = 0;
};

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

std::string QuoteArg(const std::string& argument) {
  std::string quoted = "\"";
  for (const char ch : argument) {
    if (ch == '"') {
      quoted += '\\';
    }
    quoted += ch;
  }
  quoted += '"';
  return quoted;
}

int RunCommand(const std::string& executable,
               const std::vector<std::string>& args,
               const std::string& prefix = "") {
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
      << "  feature_extractor        Extract Phoenix features into a COLMAP database\n"
      << "  feature_matcher          Match Phoenix features in a COLMAP database\n"
      << "  reconstruction           Run COLMAP mapper for sparse reconstruction\n"
      << "  automatic_reconstructor  Extract + match + reconstruct in one command\n";
}

std::vector<std::string> SliceArgs(int argc, char** argv, int start) {
  std::vector<std::string> args;
  for (int index = start; index < argc; ++index) {
    args.emplace_back(argv[index]);
  }
  return args;
}

std::vector<std::string> ReadTextLines(const std::filesystem::path& path,
                                       const bool allow_comments = false) {
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

std::vector<std::string> CollectImageNamesFromDirectory(
    const std::filesystem::path& image_path) {
  std::vector<std::string> image_names;
  for (const auto& entry : std::filesystem::directory_iterator(image_path)) {
    if (!entry.is_regular_file() ||
        !IsSupportedImageExtension(entry.path())) {
      continue;
    }
    image_names.push_back(entry.path().filename().string());
  }
  std::sort(image_names.begin(), image_names.end());
  return image_names;
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
    const auto current_path = image_path / image_names[index];
    const cv::Mat current_frame = LoadStaticFilterFrame(current_path);
    const double frame_difference =
        ComputeAdjacentFrameDifference(previous_kept_frame, current_frame);
    if (frame_difference <= static_frame_diff_threshold) {
      ++(*num_filtered_static_frames);
      spdlog::info(
          "Filter static adjacent frame: {} diff_mean={:.3f} threshold={:.3f}",
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

uint64_t PairKey(const colmap::image_t image_id1,
                 const colmap::image_t image_id2) {
  const auto min_id = static_cast<uint32_t>(std::min(image_id1, image_id2));
  const auto max_id = static_cast<uint32_t>(std::max(image_id1, image_id2));
  return (static_cast<uint64_t>(min_id) << 32) | max_id;
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

int RunFeatureExtractor(int argc, char** argv) {
  constexpr int kMaxExtractedFeatures = 5000;
  using Clock = std::chrono::steady_clock;
  std::filesystem::path image_list_path;
  int camera_mode = -1;
  int max_edge = 1600;
  int top_k = 5000;
  double scores_th = 0.2;
  double static_frame_diff_threshold = 1.0;
  bool filter_static_frames = false;
  bool skip_existing = true;

  colmap::OptionManager options;
  options.AddDatabaseOptions();
  options.AddImageOptions();
  options.AddDefaultOption("camera_mode", &camera_mode);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddFeatureExtractionOptions();
  options.AddDefaultOption("Phoenix.max_edge", &max_edge);
  options.AddDefaultOption("Phoenix.top_k", &top_k);
  options.AddDefaultOption("Phoenix.scores_th", &scores_th);
  options.AddDefaultOption("Phoenix.filter_static_frames",
                           &filter_static_frames);
  options.AddDefaultOption("Phoenix.static_frame_diff_threshold",
                           &static_frame_diff_threshold);
  options.AddDefaultOption("Phoenix.skip_existing", &skip_existing);
  options.Parse(argc, argv);

  if (top_k > kMaxExtractedFeatures) {
    spdlog::warn("Phoenix.top_k={} exceeds hard cap {}, clamping",
                 top_k,
                 kMaxExtractedFeatures);
    top_k = kMaxExtractedFeatures;
  }

  colmap::ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;
  reader_options.as_rgb = true;

  if (camera_mode >= 0) {
    ApplyCameraMode(camera_mode, &reader_options);
  }

  if (!image_list_path.empty()) {
    reader_options.image_names = ReadTextLines(image_list_path);
  }

  int num_filtered_static_frames = 0;
  if (filter_static_frames) {
    std::vector<std::string> candidate_image_names = reader_options.image_names;
    if (candidate_image_names.empty()) {
      candidate_image_names = CollectImageNamesFromDirectory(*options.image_path);
    }
    reader_options.image_names = FilterStaticAdjacentFrames(
        *options.image_path,
        candidate_image_names,
        static_frame_diff_threshold,
        &num_filtered_static_frames);
  }

  if ((filter_static_frames || !image_list_path.empty()) &&
      reader_options.image_names.empty()) {
    return EXIT_SUCCESS;
  }

  reader_options.Check();

  auto database = colmap::Database::Open(*options.database_path);
  ImportImages(reader_options, database.get());

  const std::vector<colmap::Image> images =
      CollectTargetImages(*database, reader_options.image_names);

  const auto backbone_engine = sfm_phoenix::EnsurePhoenixBackboneEngine();
  const auto sddh_engine = sfm_phoenix::EnsurePhoenixSddhEngine();

  sfm_phoenix::AlikedConfig detector_config;
  detector_config.backbone_engine = backbone_engine.string();
  detector_config.sddh_engine = sddh_engine.string();
  detector_config.max_edge = max_edge;
  detector_config.dkd.top_k = top_k;
  detector_config.dkd.n_limit = kMaxExtractedFeatures;
  detector_config.dkd.scores_th = static_cast<float>(scores_th);

  sfm_phoenix::AlikedDetector detector;
  if (!detector.init(detector_config)) {
    spdlog::error("Failed to initialize Phoenix ALIKED detector");
    return EXIT_FAILURE;
  }

  int num_extracted = 0;
  int num_skipped = 0;
  double total_detect_ms = 0.0;
  const auto extraction_start = Clock::now();
  database->BeginTransaction();
  for (const auto& image : images) {
    const auto image_id = image.ImageId();
    const bool has_keypoints = database->ExistsKeypoints(image_id);
    const bool has_descriptors = database->ExistsDescriptors(image_id);
    if (has_keypoints || has_descriptors) {
      if (skip_existing) {
        spdlog::info("Skip existing image: {}", image.Name());
        ++num_skipped;
        continue;
      }
      spdlog::error("Image already has features in database: {}",
                    image.Name());
      database->EndTransaction();
      return EXIT_FAILURE;
    }

    const std::filesystem::path image_path =
        std::filesystem::path(reader_options.image_path) / image.Name();
    const cv::Mat image_bgr =
        cv::imread(image_path.string(), cv::IMREAD_COLOR);
    if (image_bgr.empty()) {
      spdlog::error("Cannot read image: {}", image_path.string());
      database->EndTransaction();
      return EXIT_FAILURE;
    }

    const auto detect_start = Clock::now();
    const auto result = detector.detect(image_bgr);
    total_detect_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - detect_start)
        .count();
    const auto keypoints = sfm_phoenix::ToColmapKeypoints(result);
    const auto descriptors = sfm_phoenix::ToColmapDescriptors(result);

    database->WriteKeypoints(image_id, keypoints);
    database->WriteDescriptors(image_id, descriptors);
    ++num_extracted;

    spdlog::info("Extracted {} features for {}",
                 result.num_keypoints,
                 image.Name());
  }
  database->EndTransaction();

    const double extraction_wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() -
                          extraction_start)
        .count();
    const double avg_detect_ms =
      num_extracted > 0 ? total_detect_ms / num_extracted : 0.0;
    const double avg_extraction_wall_ms =
      num_extracted > 0 ? extraction_wall_ms / num_extracted : 0.0;
    const double extraction_images_per_sec =
      extraction_wall_ms > 0.0 ? (1000.0 * num_extracted) / extraction_wall_ms
                   : 0.0;

  spdlog::info("Phoenix feature extraction finished. extracted={} skipped={}",
               num_extracted,
               num_skipped);
    spdlog::info(
      "Phoenix extraction performance: detect_total_ms={:.3f} detect_avg_ms={:.3f} wall_total_ms={:.3f} wall_avg_ms={:.3f} images_per_sec={:.3f} filtered_static_frames={}",
      total_detect_ms,
      avg_detect_ms,
      extraction_wall_ms,
      avg_extraction_wall_ms,
      extraction_images_per_sec,
      num_filtered_static_frames);
  return EXIT_SUCCESS;
}

int RunFeatureMatcher(int argc, char** argv) {
  constexpr int kMaxMatchedFeatures = 4000;
    using Clock = std::chrono::steady_clock;
  std::filesystem::path pair_list_path;
  std::filesystem::path image_list_path;
  int max_edge = 1600;
  int max_matches = kMaxMatchedFeatures;
  int linear_overlap_num = 10;
  int quadratic_overlap_num = 10;
  bool skip_existing = true;
  bool overwrite_existing = false;

  colmap::OptionManager options;
  options.AddDatabaseOptions();
  options.AddDefaultOption("pair_list_path", &pair_list_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("Phoenix.max_edge", &max_edge);
  options.AddDefaultOption("Phoenix.max_matches", &max_matches);
  options.AddDefaultOption("Phoenix.linear_overlap_num", &linear_overlap_num);
  options.AddDefaultOption("Phoenix.quadratic_overlap_num",
                           &quadratic_overlap_num);
  options.AddDefaultOption("Phoenix.skip_existing", &skip_existing);
  options.AddDefaultOption("Phoenix.overwrite_existing", &overwrite_existing);
  options.Parse(argc, argv);

  if (linear_overlap_num < 0 || quadratic_overlap_num < 0) {
    spdlog::error("Phoenix overlap params must be non-negative");
    return EXIT_FAILURE;
  }

  if (max_matches > kMaxMatchedFeatures) {
    spdlog::warn("Phoenix.max_matches={} exceeds hard cap {}, clamping",
                 max_matches,
                 kMaxMatchedFeatures);
    max_matches = kMaxMatchedFeatures;
  }

  auto database = colmap::Database::Open(*options.database_path);
  const std::vector<colmap::Image> all_images = LoadImages(*database);
  std::vector<std::string> selected_image_names;
  if (!image_list_path.empty()) {
    selected_image_names = ReadTextLines(image_list_path);
  }
  const std::vector<colmap::Image> ordered_images =
      SelectOrderedImages(all_images, selected_image_names);
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
    if (linear_overlap_num > 0 || quadratic_overlap_num > 0) {
      pairs = BuildSequentialPairs(ordered_images,
                                   linear_overlap_num,
                                   quadratic_overlap_num);
    } else {
      pairs = BuildExhaustivePairs(ordered_images);
    }
  } else {
    pairs = ReadPairs(pair_list_path, image_ids);
  }

  const auto lightglue_engine = sfm_phoenix::EnsurePhoenixLightGlueEngine();

  sfm_phoenix::LightGlueConfig matcher_config;
  matcher_config.engine_path = lightglue_engine.string();
  matcher_config.max_matches = max_matches;

  sfm_phoenix::LightGlueMatcher matcher;
  if (!matcher.init(matcher_config)) {
    spdlog::error("Failed to initialize Phoenix LightGlue matcher");
    return EXIT_FAILURE;
  }

  std::unordered_map<colmap::image_t, CachedFeatures> feature_cache;
  int num_matched_pairs = 0;
  int num_skipped_pairs = 0;
  int num_zero_match_pairs = 0;
  double total_match_ms = 0.0;
  const auto matching_start = Clock::now();

  database->BeginTransaction();
  for (const auto& [image_id1, image_id2] : pairs) {
    const bool has_matches = database->ExistsMatches(image_id1, image_id2);
    const bool has_geometry =
        database->ExistsInlierMatches(image_id1, image_id2);
    if ((has_matches || has_geometry) && skip_existing &&
        !overwrite_existing) {
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

    const auto match_start = Clock::now();
    const auto matches = matcher.match(features1.keypoints.data(),
                       features1.descriptors.data(),
                       features1.num_keypoints,
                       features2.keypoints.data(),
                       features2.descriptors.data(),
                       features2.num_keypoints);
    total_match_ms +=
      std::chrono::duration<double, std::milli>(Clock::now() - match_start)
        .count();
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
    } else {
      ++num_zero_match_pairs;
    }
    ++num_matched_pairs;

    spdlog::info("Matched {} <-> {}: {}",
                 images_by_id.at(image_id1).Name(),
                 images_by_id.at(image_id2).Name(),
                 matches.num_matches);
  }
  database->EndTransaction();

    const double matching_wall_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - matching_start)
        .count();
    const double avg_match_ms =
      num_matched_pairs > 0 ? total_match_ms / num_matched_pairs : 0.0;
    const double avg_matching_wall_ms =
      num_matched_pairs > 0 ? matching_wall_ms / num_matched_pairs : 0.0;
    const double pairs_per_sec =
      matching_wall_ms > 0.0 ? (1000.0 * num_matched_pairs) / matching_wall_ms
                 : 0.0;

  spdlog::info("Phoenix matching finished. matched_pairs={} skipped_pairs={}",
               num_matched_pairs,
               num_skipped_pairs);
    spdlog::info(
      "Phoenix matching performance: scheduled_pairs={} matched_pairs={} skipped_pairs={} zero_match_pairs={} match_total_ms={:.3f} match_avg_ms={:.3f} wall_total_ms={:.3f} wall_avg_ms={:.3f} pairs_per_sec={:.3f}",
      pairs.size(),
      num_matched_pairs,
      num_skipped_pairs,
      num_zero_match_pairs,
      total_match_ms,
      avg_match_ms,
      matching_wall_ms,
      avg_matching_wall_ms,
      pairs_per_sec);
  return EXIT_SUCCESS;
}

struct AutomaticOptions {
  std::string database_path;
  std::string image_path;
  std::string output_path;
  std::string image_list_path;
  std::string pair_list_path;
  std::string colmap_path = "colmap";
  int camera_mode = -1;
  int max_edge = 1600;
  int top_k = 5000;
  int max_matches = 4000;
  double static_frame_diff_threshold = 1.0;
  int linear_overlap_num = 0;
  int quadratic_overlap_num = 0;
  bool filter_static_frames = false;
  bool skip_existing = true;
  bool overwrite_existing = false;
  std::vector<std::string> mapper_args;
};

bool ParseBool(const std::string& value) {
  return value == "1" || value == "true" || value == "TRUE" ||
         value == "True";
}

AutomaticOptions ParseAutomaticOptions(int argc, char** argv) {
  AutomaticOptions options;
  const std::unordered_set<std::string> handled_flags = {
      "--database_path",
      "--image_path",
      "--output_path",
      "--image_list_path",
      "--pair_list_path",
      "--colmap_path",
      "--camera_mode",
      "--Phoenix.max_edge",
      "--Phoenix.top_k",
      "--Phoenix.max_matches",
      "--Phoenix.filter_static_frames",
      "--Phoenix.static_frame_diff_threshold",
      "--Phoenix.linear_overlap_num",
      "--Phoenix.quadratic_overlap_num",
      "--Phoenix.skip_existing",
      "--Phoenix.overwrite_existing",
  };

  for (int index = 2; index < argc; ++index) {
    const std::string arg = argv[index];
    const bool has_value = index + 1 < argc;
    if (arg == "--database_path" && has_value) {
      options.database_path = argv[++index];
    } else if (arg == "--image_path" && has_value) {
      options.image_path = argv[++index];
    } else if (arg == "--output_path" && has_value) {
      options.output_path = argv[++index];
    } else if (arg == "--image_list_path" && has_value) {
      options.image_list_path = argv[++index];
    } else if (arg == "--pair_list_path" && has_value) {
      options.pair_list_path = argv[++index];
    } else if (arg == "--colmap_path" && has_value) {
      options.colmap_path = argv[++index];
    } else if (arg == "--camera_mode" && has_value) {
      options.camera_mode = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.max_edge" && has_value) {
      options.max_edge = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.top_k" && has_value) {
      options.top_k = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.max_matches" && has_value) {
      options.max_matches = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.filter_static_frames" && has_value) {
      options.filter_static_frames = ParseBool(argv[++index]);
    } else if (arg == "--Phoenix.static_frame_diff_threshold" && has_value) {
      options.static_frame_diff_threshold = std::stod(argv[++index]);
    } else if (arg == "--Phoenix.linear_overlap_num" && has_value) {
      options.linear_overlap_num = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.quadratic_overlap_num" && has_value) {
      options.quadratic_overlap_num = std::stoi(argv[++index]);
    } else if (arg == "--Phoenix.skip_existing" && has_value) {
      options.skip_existing = ParseBool(argv[++index]);
    } else if (arg == "--Phoenix.overwrite_existing" && has_value) {
      options.overwrite_existing = ParseBool(argv[++index]);
    } else {
      options.mapper_args.push_back(arg);
      if (arg.rfind("--", 0) == 0 && has_value &&
          handled_flags.find(arg) == handled_flags.end() &&
          std::string(argv[index + 1]).rfind("--", 0) != 0) {
        options.mapper_args.push_back(argv[++index]);
      }
    }
  }

  if (options.database_path.empty() || options.image_path.empty() ||
      options.output_path.empty()) {
    throw std::runtime_error(
        "automatic_reconstructor requires --database_path, --image_path, "
        "and --output_path");
  }
  if (options.linear_overlap_num < 0 || options.quadratic_overlap_num < 0) {
    throw std::runtime_error(
        "Phoenix overlap params must be non-negative");
  }
  if (options.static_frame_diff_threshold < 0.0) {
    throw std::runtime_error(
        "Phoenix static_frame_diff_threshold must be non-negative");
  }
  return options;
}

}  // namespace

int main(int argc, char** argv) {
  try {
    InitSpdLog();

    if (argc < 2) {
      PrintHelp();
      return EXIT_FAILURE;
    }

    const std::string command = argv[1];
    const auto exe_dir = ExecutableDirectory(argv[0]);

    if (command == "help" || command == "-h" || command == "--help") {
      PrintHelp();
      return EXIT_SUCCESS;
    }

    if (command == "feature_extractor") {
      return RunFeatureExtractor(argc - 1, argv + 1);
    }

    if (command == "feature_matcher") {
      return RunFeatureMatcher(argc - 1, argv + 1);
    }

    if (command == "reconstruction" || command == "mapper") {
      return RunCommand("colmap", SliceArgs(argc, argv, 2), "mapper");
    }

    if (command == "automatic_reconstructor") {
      const auto options = ParseAutomaticOptions(argc, argv);

      std::vector<std::string> extractor_args = {
          "--database_path", options.database_path,
          "--image_path", options.image_path,
          "--Phoenix.max_edge", std::to_string(options.max_edge),
          "--Phoenix.top_k", std::to_string(options.top_k),
          "--Phoenix.filter_static_frames",
          options.filter_static_frames ? "true" : "false",
          "--Phoenix.static_frame_diff_threshold",
          std::to_string(options.static_frame_diff_threshold),
          "--Phoenix.skip_existing",
          options.skip_existing ? "true" : "false",
      };
      if (!options.image_list_path.empty()) {
        extractor_args.insert(extractor_args.end(),
                              {"--image_list_path", options.image_list_path});
      }
      if (options.camera_mode >= 0) {
        extractor_args.insert(extractor_args.end(),
                              {"--camera_mode",
                               std::to_string(options.camera_mode)});
      }

      std::vector<std::string> extractor_argv = {"feature_extractor"};
      extractor_argv.insert(extractor_argv.end(),
                            extractor_args.begin(),
                            extractor_args.end());
      std::vector<char*> extractor_argp;
      extractor_argp.reserve(extractor_argv.size());
      for (auto& arg : extractor_argv) {
        extractor_argp.push_back(arg.data());
      }

      int exit_code =
          RunFeatureExtractor(static_cast<int>(extractor_argp.size()),
                              extractor_argp.data());
      if (exit_code != 0) {
        return exit_code;
      }

      std::vector<std::string> matcher_args = {
          "--database_path", options.database_path,
          "--Phoenix.max_edge", std::to_string(options.max_edge),
          "--Phoenix.max_matches", std::to_string(options.max_matches),
          "--Phoenix.linear_overlap_num",
          std::to_string(options.linear_overlap_num),
          "--Phoenix.quadratic_overlap_num",
          std::to_string(options.quadratic_overlap_num),
          "--Phoenix.skip_existing",
          options.skip_existing ? "true" : "false",
          "--Phoenix.overwrite_existing",
          options.overwrite_existing ? "true" : "false",
      };
      if (!options.image_list_path.empty()) {
        matcher_args.insert(matcher_args.end(),
                            {"--image_list_path", options.image_list_path});
      }
      if (!options.pair_list_path.empty()) {
        matcher_args.insert(matcher_args.end(),
                            {"--pair_list_path", options.pair_list_path});
      }

      std::vector<std::string> matcher_argv = {"feature_matcher"};
      matcher_argv.insert(matcher_argv.end(),
                          matcher_args.begin(),
                          matcher_args.end());
      std::vector<char*> matcher_argp;
      matcher_argp.reserve(matcher_argv.size());
      for (auto& arg : matcher_argv) {
        matcher_argp.push_back(arg.data());
      }

      exit_code = RunFeatureMatcher(static_cast<int>(matcher_argp.size()),
                                    matcher_argp.data());
      if (exit_code != 0) {
        return exit_code;
      }

      std::vector<std::string> mapper_args = {
          "--database_path", options.database_path,
          "--image_path", options.image_path,
          "--output_path", options.output_path,
      };
      mapper_args.insert(mapper_args.end(),
                         options.mapper_args.begin(),
                         options.mapper_args.end());
      std::filesystem::create_directories(options.output_path);
      return RunCommand(options.colmap_path, mapper_args, "mapper");
    }

    PrintHelp();
    return EXIT_FAILURE;
  } catch (const std::exception& error) {
    std::cerr << error.what() << std::endl;
    return EXIT_FAILURE;
  }
}