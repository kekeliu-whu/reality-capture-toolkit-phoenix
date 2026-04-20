#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/model_runtime.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/sensor/rig.h>

#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <unordered_map>

namespace phoenix_tool {

namespace {

struct ImportedCameraEntry {
  colmap::camera_t camera_id = colmap::kInvalidCameraId;
  colmap::rig_t rig_id = colmap::kInvalidRigId;
  size_t width = 0;
  size_t height = 0;
};

std::string CameraGroupKey(const colmap::ImageReaderOptions& reader_options,
                           const std::string& image_name,
                           const bool default_share_camera_per_folder) {
  if (reader_options.existing_camera_id != colmap::kInvalidCameraId) {
    return "__existing_camera__";
  }
  if (reader_options.single_camera) {
    return "__single_camera__";
  }
  if (reader_options.single_camera_per_folder ||
      default_share_camera_per_folder) {
    return std::filesystem::path(image_name).parent_path().string();
  }
  return image_name;
}

colmap::Rig EnsureRigForCamera(colmap::Database* database,
                               const colmap::Camera& camera) {
  if (std::optional<colmap::Rig> rig =
          database->ReadRigWithSensor(camera.SensorId());
      rig.has_value()) {
    return *rig;
  }

  colmap::Rig rig;
  rig.AddRefSensor(camera.SensorId());
  rig.SetRigId(database->WriteRig(rig));
  return rig;
}

ImportedCameraEntry ResolveCameraForImage(
    const colmap::ImageReaderOptions& reader_options,
    const std::string& image_name,
    const size_t width,
    const size_t height,
    const bool default_share_camera_per_folder,
    colmap::Database* database,
    std::unordered_map<std::string, ImportedCameraEntry>* camera_cache) {
  const std::string cache_key = CameraGroupKey(
      reader_options, image_name, default_share_camera_per_folder);
  const auto cached = camera_cache->find(cache_key);
  if (cached != camera_cache->end()) {
    if (cached->second.width != width || cached->second.height != height) {
      throw std::runtime_error("Camera dimensions mismatch for image: " +
                               image_name);
    }
    return cached->second;
  }

  colmap::Camera camera;
  if (reader_options.existing_camera_id != colmap::kInvalidCameraId) {
    camera = database->ReadCamera(reader_options.existing_camera_id);
    if (camera.width != width || camera.height != height) {
      throw std::runtime_error("Existing camera dimensions mismatch for image: " +
                               image_name);
    }
  } else {
    const double focal_length =
        reader_options.default_focal_length_factor *
        static_cast<double>(std::max(width, height));
    camera = colmap::Camera::CreateFromModelName(
        colmap::kInvalidCameraId,
        reader_options.camera_model,
        focal_length,
        width,
        height);
    if (!reader_options.camera_params.empty()) {
      if (!camera.SetParamsFromString(reader_options.camera_params)) {
        throw std::runtime_error("Invalid camera_params for image: " +
                                 image_name);
      }
      camera.has_prior_focal_length = true;
    }
    if (!camera.VerifyParams()) {
      throw std::runtime_error("Invalid camera parameters for image: " +
                               image_name);
    }
    camera.camera_id = database->WriteCamera(camera);
  }

  const colmap::Rig rig = EnsureRigForCamera(database, camera);
  const ImportedCameraEntry entry = {
      camera.camera_id, rig.RigId(), width, height};
  camera_cache->emplace(cache_key, entry);
  return entry;
}

void LogFeatureExtractionMetrics(const FeatureExtractionMetrics& metrics) {
  const double images_per_sec =
      metrics.wall_total_ms > 0.0
          ? (1000.0 * metrics.num_extracted) / metrics.wall_total_ms
          : 0.0;

  spdlog::info(
      "Phoenix extraction performance: candidate_images={} "
      "selected_images={} extracted={} skipped={} filtered_static_frames={} "
      "static_filter_total_ms={:.3f} import_images_ms={:.3f} "
      "image_registration_total_ms={:.3f} feature_check_total_ms={:.3f} "
      "engine_init_ms={:.3f} image_decode_total_ms={:.3f} "
      "image_decode_avg_ms={:.3f} image_decode_max_ms={:.3f} "
      "detect_total_ms={:.3f} detect_avg_ms={:.3f} detect_max_ms={:.3f} "
      "convert_total_ms={:.3f} convert_avg_ms={:.3f} convert_max_ms={:.3f} "
      "db_write_total_ms={:.3f} db_write_avg_ms={:.3f} db_write_max_ms={:.3f} "
      "wall_total_ms={:.3f} wall_avg_ms={:.3f} images_per_sec={:.3f}",
      metrics.candidate_images,
      metrics.selected_images,
      metrics.num_extracted,
      metrics.num_skipped,
      metrics.num_filtered_static_frames,
      metrics.static_filter_ms.total_ms,
      metrics.import_images_ms.total_ms,
      metrics.image_registration_ms.total_ms,
      metrics.feature_check_ms.total_ms,
      metrics.engine_init_ms.total_ms,
      metrics.image_decode_ms.total_ms,
      metrics.image_decode_ms.AverageMs(),
      metrics.image_decode_ms.max_ms,
      metrics.detect_ms.total_ms,
      metrics.detect_ms.AverageMs(),
      metrics.detect_ms.max_ms,
      metrics.convert_ms.total_ms,
      metrics.convert_ms.AverageMs(),
      metrics.convert_ms.max_ms,
      metrics.db_write_ms.total_ms,
      metrics.db_write_ms.AverageMs(),
      metrics.db_write_ms.max_ms,
      metrics.wall_total_ms,
      metrics.num_extracted > 0 ? metrics.wall_total_ms / metrics.num_extracted
                                : 0.0,
      images_per_sec);
}

}  // namespace

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

  FeatureExtractionMetrics metrics;
  const auto extraction_start = Clock::now();

  if (!image_list_path.empty()) {
    reader_options.image_names = ReadTextLines(image_list_path);
  }

  std::vector<std::string> candidate_image_names = reader_options.image_names;
  if (candidate_image_names.empty()) {
    candidate_image_names = CollectImageNamesFromDirectory(*options.image_path);
  }
  metrics.candidate_images = static_cast<int>(candidate_image_names.size());

  if (filter_static_frames) {
    const auto static_filter_start = Clock::now();
    reader_options.image_names = FilterStaticAdjacentFrames(
        *options.image_path,
        candidate_image_names,
        static_frame_diff_threshold,
        &metrics.num_filtered_static_frames);
    metrics.static_filter_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  static_filter_start)
            .count());
  } else {
    reader_options.image_names = std::move(candidate_image_names);
  }

  metrics.selected_images = static_cast<int>(reader_options.image_names.size());
  if ((filter_static_frames || !image_list_path.empty()) &&
      reader_options.image_names.empty()) {
    metrics.wall_total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  extraction_start)
            .count();
    LogFeatureExtractionMetrics(metrics);
    return EXIT_SUCCESS;
  }

  reader_options.Check();

  auto database = colmap::Database::Open(*options.database_path);
  const bool default_share_camera_per_folder =
      camera_mode < 0 &&
      reader_options.existing_camera_id == colmap::kInvalidCameraId &&
      !reader_options.single_camera &&
      !reader_options.single_camera_per_folder &&
      !reader_options.single_camera_per_image;

  const auto engine_init_start = Clock::now();
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
  metrics.engine_init_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                engine_init_start)
          .count());

  std::vector<std::string> existing_image_names;
  existing_image_names.reserve(reader_options.image_names.size());
  std::unordered_map<std::string, ImportedCameraEntry> camera_cache;

  database->BeginTransaction();
  for (const auto& image_name : reader_options.image_names) {
    if (database->ExistsImageWithName(image_name)) {
      existing_image_names.push_back(image_name);
      continue;
    }

    const std::filesystem::path image_path =
        std::filesystem::path(reader_options.image_path) / image_name;

    const auto decode_start = Clock::now();
    const cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    metrics.image_decode_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count());
    if (image_bgr.empty()) {
      spdlog::error("Cannot read image: {}", image_path.string());
      return EXIT_FAILURE;
    }

    const auto import_start = Clock::now();
    const ImportedCameraEntry camera_entry = ResolveCameraForImage(
        reader_options,
        image_name,
        static_cast<size_t>(image_bgr.cols),
        static_cast<size_t>(image_bgr.rows),
      default_share_camera_per_folder,
        database.get(),
        &camera_cache);
    metrics.import_images_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - import_start)
            .count());

    colmap::Image image;
    image.SetName(image_name);
    image.SetCameraId(camera_entry.camera_id);

    const auto registration_start = Clock::now();
    const auto image_id = database->WriteImage(image);
    image.SetImageId(image_id);
    colmap::Frame frame;
    frame.SetRigId(camera_entry.rig_id);
    frame.AddDataId(image.DataId());
    database->WriteFrame(frame);
    metrics.image_registration_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  registration_start)
            .count());

    const auto feature_check_start = Clock::now();
    const bool has_keypoints = database->ExistsKeypoints(image_id);
    const bool has_descriptors = database->ExistsDescriptors(image_id);
    metrics.feature_check_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  feature_check_start)
            .count());
    if (has_keypoints || has_descriptors) {
      if (skip_existing) {
        spdlog::info("Skip existing image: {}", image.Name());
        ++metrics.num_skipped;
        continue;
      }
      spdlog::error("Image already has features in database: {}",
                    image.Name());
      return EXIT_FAILURE;
    }

    const auto detect_start = Clock::now();
    const auto result = detector.detect(image_bgr);
    metrics.detect_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - detect_start)
            .count());

    const auto convert_start = Clock::now();
    const auto keypoints = sfm_phoenix::ToColmapKeypoints(result);
    const auto descriptors = sfm_phoenix::ToColmapDescriptors(result);
    metrics.convert_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  convert_start)
            .count());

    const auto db_write_start = Clock::now();
    database->WriteKeypoints(image_id, keypoints);
    database->WriteDescriptors(image_id, descriptors);
    metrics.db_write_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  db_write_start)
            .count());
    ++metrics.num_extracted;

    spdlog::info("Extracted {} features for {}",
                 result.num_keypoints,
                 image.Name());
  }
  database->EndTransaction();

  if (!existing_image_names.empty()) {
    const std::vector<colmap::Image> existing_images =
        CollectTargetImages(*database, existing_image_names);

    database->BeginTransaction();
    for (const auto& image : existing_images) {
      const auto image_id = image.ImageId();
      const auto feature_check_start = Clock::now();
      const bool has_keypoints = database->ExistsKeypoints(image_id);
      const bool has_descriptors = database->ExistsDescriptors(image_id);
      metrics.feature_check_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() -
                                                    feature_check_start)
              .count());
      if (has_keypoints || has_descriptors) {
        if (skip_existing) {
          spdlog::info("Skip existing image: {}", image.Name());
          ++metrics.num_skipped;
          continue;
        }
        spdlog::error("Image already has features in database: {}",
                      image.Name());
        database->EndTransaction();
        return EXIT_FAILURE;
      }

      const std::filesystem::path image_path =
          std::filesystem::path(reader_options.image_path) / image.Name();

      const auto decode_start = Clock::now();
      const cv::Mat image_bgr =
          cv::imread(image_path.string(), cv::IMREAD_COLOR);
      metrics.image_decode_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
              .count());
      if (image_bgr.empty()) {
        spdlog::error("Cannot read image: {}", image_path.string());
        database->EndTransaction();
        return EXIT_FAILURE;
      }

      const auto detect_start = Clock::now();
      const auto result = detector.detect(image_bgr);
      metrics.detect_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() - detect_start)
              .count());

      const auto convert_start = Clock::now();
      const auto keypoints = sfm_phoenix::ToColmapKeypoints(result);
      const auto descriptors = sfm_phoenix::ToColmapDescriptors(result);
      metrics.convert_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() -
                                                    convert_start)
              .count());

      const auto db_write_start = Clock::now();
      database->WriteKeypoints(image_id, keypoints);
      database->WriteDescriptors(image_id, descriptors);
      metrics.db_write_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() -
                                                    db_write_start)
              .count());
      ++metrics.num_extracted;

      spdlog::info("Extracted {} features for {}",
                   result.num_keypoints,
                   image.Name());
    }
    database->EndTransaction();
  }

  metrics.wall_total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - extraction_start)
          .count();

  spdlog::info("Phoenix feature extraction finished. extracted={} skipped={}",
               metrics.num_extracted,
               metrics.num_skipped);
  LogFeatureExtractionMetrics(metrics);
  return EXIT_SUCCESS;
}

}  // namespace phoenix_tool