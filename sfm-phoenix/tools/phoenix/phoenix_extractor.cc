#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/model_runtime.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/sensor/rig.h>

#include <oneapi/tbb/parallel_pipeline.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <cuda_runtime.h>

#include <chrono>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace phoenix_tool {

namespace {

using Clock = std::chrono::steady_clock;

constexpr int kPhoenixFixedBatchSize = 1;

int ClampSupportedPhoenixBatchSize(const int requested,
                                   const int max_supported) {
  constexpr int kSupportedBatchSizes[] = {8, 4, 1};
  for (const int batch_size : kSupportedBatchSizes) {
    if (batch_size <= requested && batch_size <= max_supported) {
      return batch_size;
    }
  }
  return 1;
}

size_t EstimateBackboneEngineReserveBytes(const int batch_size) {
  switch (ClampSupportedPhoenixBatchSize(batch_size, /*max_supported=*/8)) {
    case 8:
      return static_cast<size_t>(7300) * 1024 * 1024;
    case 4:
      return static_cast<size_t>(5600) * 1024 * 1024;
    default:
      return static_cast<size_t>(4300) * 1024 * 1024;
  }
}

struct ImportedCameraEntry {
  colmap::camera_t camera_id = colmap::kInvalidCameraId;
  colmap::rig_t rig_id = colmap::kInvalidRigId;
  size_t width = 0;
  size_t height = 0;
};

struct DecodedImageTask {
  std::string image_name;
  std::filesystem::path image_path;
  std::vector<uint8_t> encoded_bytes;
  cv::Mat image_bgr;
  double file_read_ms = 0.0;
  double image_parse_ms = 0.0;
  double decode_ms = 0.0;
};

// FeatureExtractionOptions is defined in phoenix_tool.h.

struct ExtractorExecutionContext {
  const colmap::ImageReaderOptions& reader_options;
  bool default_share_camera_per_folder = false;
  bool skip_existing = true;
  int detect_batch_size = 1;
  colmap::Database* database = nullptr;
  sfm_phoenix::AlikedDetector* detector = nullptr;
  FeatureExtractionMetrics* metrics = nullptr;
  std::unordered_map<std::string, ImportedCameraEntry>* camera_cache =
      nullptr;
};

enum class FeatureAvailability {
  kReady,
  kSkip,
  kConflict,
};

size_t DecodePipelineTokens() {
  const unsigned int hw = std::thread::hardware_concurrency();
  const size_t concurrency = hw == 0 ? 4 : static_cast<size_t>(hw);
  return std::max<size_t>(2, std::min<size_t>(8, concurrency));
}

std::vector<uint8_t> ReadFileBytes(const std::filesystem::path& image_path) {
  std::ifstream stream(image_path, std::ios::binary | std::ios::ate);
  if (!stream.is_open()) {
    throw std::runtime_error("Cannot open image file: " + image_path.string());
  }

  const std::streamsize file_size = stream.tellg();
  if (file_size <= 0) {
    throw std::runtime_error("Image file is empty: " + image_path.string());
  }
  stream.seekg(0, std::ios::beg);

  std::vector<uint8_t> bytes(static_cast<size_t>(file_size));
  if (!stream.read(reinterpret_cast<char*>(bytes.data()), file_size)) {
    throw std::runtime_error("Cannot read image file: " + image_path.string());
  }
  return bytes;
}

void AddFeatureExtractorOptions(colmap::OptionManager* options,
                                FeatureExtractionOptions* cli_options) {
  options->AddDatabaseOptions();
  options->AddImageOptions();
  options->AddDefaultOption("camera_mode", &cli_options->camera_mode);
  options->AddFeatureExtractionOptions();
  options->AddDefaultOption("image_list_path", &cli_options->image_list_path);
  options->AddDefaultOption("Phoenix.aliked_model_path",
                            &cli_options->aliked_model_path);
  options->AddDefaultOption("Phoenix.max_edge", &cli_options->max_edge);
  options->AddDefaultOption("Phoenix.top_k", &cli_options->top_k);
  options->AddDefaultOption("Phoenix.scores_th", &cli_options->scores_th);
  options->AddDefaultOption("Phoenix.single_camera_per_folder",
                            &cli_options->single_camera_per_folder);
  options->AddDefaultOption("Phoenix.filter_static_frames",
                            &cli_options->filter_static_frames);
  options->AddDefaultOption("Phoenix.static_frame_diff_threshold",
                            &cli_options->static_frame_diff_threshold);
  options->AddDefaultOption("Phoenix.skip_existing",
                            &cli_options->skip_existing);
  options->AddDefaultOption("Phoenix.detect_batch_size",
                            &cli_options->detect_batch_size);
}

void ClampFeatureExtractionOptions(FeatureExtractionOptions* cli_options) {
  constexpr int kMaxExtractedFeatures = 5000;
  if (cli_options->top_k > kMaxExtractedFeatures) {
    spdlog::warn("Phoenix.top_k={} exceeds hard cap {}, clamping",
                 cli_options->top_k,
                 kMaxExtractedFeatures);
    cli_options->top_k = kMaxExtractedFeatures;
  }
  cli_options->detect_batch_size = kPhoenixFixedBatchSize;
}

// Builds ImageReaderOptions from a raw image_path (no COLMAP OptionManager).
// Used by ExecFeatureExtractor when called from the automatic pipeline.
colmap::ImageReaderOptions BuildReaderOptionsFromPath(
    const std::filesystem::path& image_path,
    const FeatureExtractionOptions& opts) {
  colmap::ImageReaderOptions reader_options;
  reader_options.image_path = image_path.string();
  reader_options.as_rgb = true;
  if (!opts.image_list_path.empty()) {
    reader_options.image_names =
        ReadTextLines(opts.image_list_path, /*allow_comments=*/true);
  }

  if (!opts.camera_model.empty()) {
    reader_options.camera_model = opts.camera_model;
  }
  if (!opts.camera_params.empty()) {
    reader_options.camera_params = opts.camera_params;
  }
  if (opts.camera_mode >= 0) {
    ApplyCameraMode(opts.camera_mode, &reader_options);
  } else if (opts.single_camera_per_folder) {
    reader_options.single_camera = false;
    reader_options.single_camera_per_folder = true;
    reader_options.single_camera_per_image = false;
  }

  return reader_options;
}

// Builds ImageReaderOptions using COLMAP OptionManager (for CLI path).
colmap::ImageReaderOptions BuildReaderOptions(
    const colmap::OptionManager& options,
    const FeatureExtractionOptions& cli_options) {
  colmap::ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;
  reader_options.as_rgb = true;
  if (!cli_options.image_list_path.empty()) {
    reader_options.image_names =
        ReadTextLines(cli_options.image_list_path, /*allow_comments=*/true);
  }

  if (cli_options.camera_mode >= 0) {
    ApplyCameraMode(cli_options.camera_mode, &reader_options);
  } else if (cli_options.single_camera_per_folder) {
    reader_options.single_camera = false;
    reader_options.single_camera_per_folder = true;
    reader_options.single_camera_per_image = false;
  }

  return reader_options;
}

void SelectImagesForExtraction(
    const FeatureExtractionOptions& cli_options,
    const std::filesystem::path& image_path,
    colmap::ImageReaderOptions* reader_options,
    FeatureExtractionMetrics* metrics) {
  std::vector<std::string> candidate_image_names = reader_options->image_names;
  if (candidate_image_names.empty()) {
    candidate_image_names = CollectImageNamesFromDirectory(image_path, true);
  }
  metrics->candidate_images = static_cast<int>(candidate_image_names.size());

  if (cli_options.filter_static_frames) {
    const auto static_filter_start = Clock::now();
    reader_options->image_names = FilterStaticAdjacentFrames(
        image_path,
        candidate_image_names,
        cli_options.static_frame_diff_threshold,
        &metrics->num_filtered_static_frames);
    metrics->static_filter_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  static_filter_start)
            .count());
  } else {
    reader_options->image_names = std::move(candidate_image_names);
  }

  metrics->selected_images = static_cast<int>(reader_options->image_names.size());
}

bool ShouldExitEarlyAfterSelection(
    const FeatureExtractionOptions& cli_options,
    const colmap::ImageReaderOptions& reader_options) {
  return cli_options.filter_static_frames &&
         reader_options.image_names.empty();
}

bool DefaultShareCameraPerFolder(
    const FeatureExtractionOptions& cli_options,
    const colmap::ImageReaderOptions& reader_options) {
  return cli_options.camera_mode < 0 &&
      cli_options.single_camera_per_folder &&
         reader_options.existing_camera_id == colmap::kInvalidCameraId &&
         !reader_options.single_camera &&
         !reader_options.single_camera_per_folder &&
         !reader_options.single_camera_per_image;
}

// Estimate the largest supported extraction batch whose engine/context reserve
// plus per-image activations still fit in current free VRAM.
int EstimateDetectBatchSize(int max_edge, int descriptor_dim,
                            int engine_max_batch) {
  size_t free_bytes = 0, total_bytes = 0;
  if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
    spdlog::warn("cudaMemGetInfo failed, defaulting batch_size=1");
    return 1;
  }
  // Padded dimensions (rounded up to 32).
  const int pad = ((max_edge + 31) / 32) * 32;
  // Per-image backbone GPU memory:
  //   input:       3 * H * W * 4 bytes
  //   feature_map: C * H * W * 4 bytes
  //   score_map:   1 * H * W * 4 bytes
  const size_t per_image =
      static_cast<size_t>(3 + descriptor_dim + 1) * pad * pad * sizeof(float);
  const size_t cuda_safety_bytes = static_cast<size_t>(384) * 1024 * 1024;
  int batch = 1;
  for (const int candidate : {8, 4, 1}) {
    if (candidate > engine_max_batch) {
      continue;
    }
    const size_t required_bytes = EstimateBackboneEngineReserveBytes(candidate) +
                                  per_image * candidate + cuda_safety_bytes;
    if (free_bytes >= required_bytes) {
      batch = candidate;
      break;
    }
  }
  spdlog::info("GPU memory: free={:.0f} MB  total={:.0f} MB  "
               "per_image~={:.1f} MB  auto batch_size={}",
               free_bytes / 1e6, total_bytes / 1e6,
               per_image / 1e6, batch);
  return batch;
}

bool InitializeDetector(const FeatureExtractionOptions& cli_options,
             const int detect_batch_size,
                       FeatureExtractionMetrics* metrics,
                       sfm_phoenix::AlikedDetector* detector) {
  constexpr int kMaxExtractedFeatures = 5000;
  (void)detect_batch_size;
  const ResourceSnapshot before_init = CaptureResourceSnapshot();
  LogResourceSnapshot("extract.engine.before_init", before_init);
  const auto engine_init_start = Clock::now();

  sfm_phoenix::AlikedConfig detector_config;
  if (cli_options.aliked_model_path.empty()) {
    spdlog::error("Phoenix.aliked_model_path must be set");
    return false;
  }
  detector_config.full_model_path = cli_options.aliked_model_path;
  detector_config.max_edge = cli_options.max_edge;
  detector_config.dkd.top_k = cli_options.top_k;
  detector_config.dkd.n_limit = kMaxExtractedFeatures;
  detector_config.dkd.scores_th = static_cast<float>(cli_options.scores_th);

  if (!detector->init(detector_config)) {
    spdlog::error("Failed to initialize Phoenix ALIKED detector");
    return false;
  }

  metrics->engine_init_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                engine_init_start)
          .count());
  const ResourceSnapshot after_init = CaptureResourceSnapshot();
  LogResourceSnapshot("extract.engine.after_init",
                      after_init,
                      &before_init,
                      "full_model=" +
                          std::filesystem::path(cli_options.aliked_model_path)
                              .filename()
                              .string());
  return true;
}

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

FeatureAvailability CheckFeatureAvailability(colmap::Database* database,
                                             const colmap::image_t image_id,
                                             const std::string& image_name,
                                             const bool skip_existing,
                                             FeatureExtractionMetrics* metrics) {
  const auto feature_check_start = Clock::now();
  const bool has_keypoints = database->ExistsKeypoints(image_id);
  const bool has_descriptors = database->ExistsDescriptors(image_id);
  metrics->feature_check_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                feature_check_start)
          .count());

  if (!has_keypoints && !has_descriptors) {
    return FeatureAvailability::kReady;
  }

  if (skip_existing) {
    spdlog::info("Skip existing image: {}", image_name);
    ++metrics->num_skipped;
    return FeatureAvailability::kSkip;
  }

  return FeatureAvailability::kConflict;
}

colmap::image_t RegisterImageAndFrame(const std::string& image_name,
                                      const ImportedCameraEntry& camera_entry,
                                      colmap::Database* database,
                                      FeatureExtractionMetrics* metrics) {
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

  metrics->image_registration_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                registration_start)
          .count());
  return image_id;
}

void StoreDetectedFeatures(const std::string& image_name,
                           const sfm_phoenix::AlikedResult& result,
                           const colmap::image_t image_id,
                           colmap::Database* database,
                           FeatureExtractionMetrics* metrics) {
  const auto convert_start = Clock::now();
  const auto keypoints = sfm_phoenix::ToColmapKeypoints(result);
  const auto descriptors = sfm_phoenix::ToColmapDescriptors(result);
  metrics->convert_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() - convert_start)
          .count());

  const auto db_write_start = Clock::now();
  database->WriteKeypoints(image_id, keypoints);
  database->WriteDescriptors(image_id, descriptors);
  metrics->db_write_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() - db_write_start)
          .count());
  ++metrics->num_extracted;

  spdlog::info("Extracted {} features for {}",
               result.num_keypoints,
               image_name);
}

void DetectAndStoreFeatures(const std::string& image_name,
                            const cv::Mat& image_bgr,
                            const colmap::image_t image_id,
                            sfm_phoenix::AlikedDetector* detector,
                            colmap::Database* database,
                            FeatureExtractionMetrics* metrics) {
  const auto detect_start = Clock::now();
  const auto result = detector->detect(image_bgr);
  metrics->detect_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() - detect_start)
          .count());
  StoreDetectedFeatures(image_name, result, image_id, database, metrics);
}

void ProcessNewImages(
    const std::vector<std::string>& image_names,
    const colmap::ImageReaderOptions& reader_options,
    ExtractorExecutionContext* context,
    std::vector<std::string>* existing_image_names) {
  size_t next_image_index = 0;
  const size_t decode_tokens = DecodePipelineTokens();
  const int batch_size = context->detect_batch_size;
  ResourceSnapshot previous_snapshot = CaptureResourceSnapshot();

  // Accumulation buffer: filled by the serial stage-4 filter,
  // flushed when batch_size is reached or the pipeline ends.
  std::vector<DecodedImageTask> pending;
  pending.reserve(batch_size);

  const auto flush_pending = [&]() {
    if (pending.empty()) return;

    std::vector<cv::Mat> imgs;
    imgs.reserve(pending.size());
    for (const auto& t : pending) imgs.push_back(t.image_bgr);

    const ResourceSnapshot batch_before = CaptureResourceSnapshot();
    LogResourceSnapshot(
      "extract.batch.before_detect",
      batch_before,
      &previous_snapshot,
      "batch_size=" + std::to_string(pending.size()) +
        " first=" + pending.front().image_name +
        " last=" + pending.back().image_name);

    const auto detect_start = Clock::now();
    auto batch_results = context->detector->detect_batch(imgs);
    const double batch_ms =
        std::chrono::duration<double, std::milli>(Clock::now() - detect_start)
            .count();
    const double per_img_ms =
        batch_ms / static_cast<double>(pending.size());
    const ResourceSnapshot batch_after = CaptureResourceSnapshot();
    LogResourceSnapshot("extract.batch.after_detect",
              batch_after,
              &batch_before,
              "batch_size=" + std::to_string(pending.size()) +
                " detect_ms=" + std::to_string(batch_ms));

    for (int i = 0; i < static_cast<int>(pending.size()); ++i) {
      const DecodedImageTask& task = pending[i];
      context->metrics->file_read_ms.AddSample(task.file_read_ms);
      context->metrics->image_parse_ms.AddSample(task.image_parse_ms);
      context->metrics->image_decode_ms.AddSample(task.decode_ms);
      context->metrics->detect_ms.AddSample(per_img_ms);

      const auto import_start = Clock::now();
      const ImportedCameraEntry camera_entry = ResolveCameraForImage(
          context->reader_options,
          task.image_name,
          static_cast<size_t>(task.image_bgr.cols),
          static_cast<size_t>(task.image_bgr.rows),
          context->default_share_camera_per_folder,
          context->database,
          context->camera_cache);
      context->metrics->import_images_ms.AddSample(
          std::chrono::duration<double, std::milli>(Clock::now() - import_start)
              .count());

      const colmap::image_t image_id = RegisterImageAndFrame(
          task.image_name, camera_entry,
          context->database, context->metrics);

      const FeatureAvailability availability = CheckFeatureAvailability(
          context->database, image_id, task.image_name,
          context->skip_existing, context->metrics);
      if (availability == FeatureAvailability::kSkip) continue;
      if (availability == FeatureAvailability::kConflict) {
        throw std::runtime_error(
            "Image already has features in database: " + task.image_name);
      }

      StoreDetectedFeatures(task.image_name, batch_results[i],
                            image_id, context->database, context->metrics);
      const ResourceSnapshot current_snapshot = CaptureResourceSnapshot();
      LogResourceSnapshot("extract.image.done",
                          current_snapshot,
                          &previous_snapshot,
                          "image=" + task.image_name +
                              " keypoints=" +
                              std::to_string(batch_results[i].num_keypoints));
      previous_snapshot = current_snapshot;
    }
    pending.clear();
  };

  context->database->BeginTransaction();
  tbb::parallel_pipeline(
      decode_tokens,
      tbb::make_filter<void, DecodedImageTask>(
          tbb::filter_mode::serial_in_order,
          [&](tbb::flow_control& control) -> DecodedImageTask {
            if (IsInterruptRequested()) {
              control.stop();
              return {};
            }
            while (next_image_index < image_names.size()) {
              const std::string& image_name = image_names[next_image_index++];
              if (context->database->ExistsImageWithName(image_name)) {
                existing_image_names->push_back(image_name);
                continue;
              }

              DecodedImageTask task;
              task.image_name = image_name;
              task.image_path =
                  std::filesystem::path(reader_options.image_path) / image_name;
              return task;
            }

            control.stop();
            return {};
          }) &
          tbb::make_filter<DecodedImageTask, DecodedImageTask>(
              tbb::filter_mode::parallel,
              [&](DecodedImageTask task) -> DecodedImageTask {
                const auto read_start = Clock::now();
                task.encoded_bytes = ReadFileBytes(task.image_path);
                task.file_read_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() -
                                                              read_start)
                        .count();
                return task;
              }) &
          tbb::make_filter<DecodedImageTask, DecodedImageTask>(
              tbb::filter_mode::parallel,
              [&](DecodedImageTask task) -> DecodedImageTask {
                const auto parse_start = Clock::now();
                task.image_bgr =
                    cv::imdecode(task.encoded_bytes, cv::IMREAD_COLOR);
                task.image_parse_ms =
                    std::chrono::duration<double, std::milli>(Clock::now() -
                                                              parse_start)
                        .count();
                task.decode_ms = task.file_read_ms + task.image_parse_ms;
                if (task.image_bgr.empty()) {
                  throw std::runtime_error("Cannot read image: " +
                                           task.image_path.string());
                }
                task.encoded_bytes.clear();
                task.encoded_bytes.shrink_to_fit();
                return task;
              }) &
          tbb::make_filter<DecodedImageTask, void>(
              tbb::filter_mode::serial_in_order,
              [&](const DecodedImageTask& task) {
                pending.push_back(task);
                if (static_cast<int>(pending.size()) >= batch_size) {
                  flush_pending();
                }
              }));
  flush_pending();  // drain remainder
  context->database->EndTransaction();
}

bool ProcessExistingImages(const std::vector<colmap::Image>& existing_images,
                          const std::filesystem::path& image_root,
                          const bool skip_existing,
                          colmap::Database* database,
                          sfm_phoenix::AlikedDetector* detector,
                          FeatureExtractionMetrics* metrics) {
  if (existing_images.empty()) {
    return true;
  }

  database->BeginTransaction();
  ResourceSnapshot previous_snapshot = CaptureResourceSnapshot();
  for (const auto& image : existing_images) {
    if (IsInterruptRequested()) {
      database->EndTransaction();
      return false;
    }
    const FeatureAvailability availability = CheckFeatureAvailability(
        database, image.ImageId(), image.Name(), skip_existing, metrics);
    if (availability == FeatureAvailability::kSkip) {
      continue;
    }
    if (availability == FeatureAvailability::kConflict) {
      spdlog::error("Image already has features in database: {}",
                    image.Name());
      database->EndTransaction();
      return false;
    }

    const std::filesystem::path image_path = image_root / image.Name();

    const auto decode_start = Clock::now();
    const cv::Mat image_bgr = cv::imread(image_path.string(), cv::IMREAD_COLOR);
    metrics->image_decode_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - decode_start)
            .count());
    if (image_bgr.empty()) {
      spdlog::error("Cannot read image: {}", image_path.string());
      database->EndTransaction();
      return false;
    }

    DetectAndStoreFeatures(
        image.Name(), image_bgr, image.ImageId(), detector, database, metrics);
    const ResourceSnapshot current_snapshot = CaptureResourceSnapshot();
    LogResourceSnapshot("extract.image.done",
                        current_snapshot,
                        &previous_snapshot,
                        "image=" + image.Name() + " existing=1");
    previous_snapshot = current_snapshot;
  }
  database->EndTransaction();
  return true;
}

void LogFeatureExtractionMetrics(const FeatureExtractionMetrics& metrics) {
  const double images_per_sec =
      metrics.wall_total_ms > 0.0
          ? (1000.0 * metrics.num_extracted) / metrics.wall_total_ms
          : 0.0;

  spdlog::info(
      "Phoenix extraction performance: candidate_images={} "
      "selected_images={} extracted={} skipped={} filtered_static_frames={} "
      "static_filter_total_ms={:.3f} file_read_total_ms={:.3f} "
      "file_read_avg_ms={:.3f} file_read_max_ms={:.3f} "
      "image_parse_total_ms={:.3f} image_parse_avg_ms={:.3f} "
      "image_parse_max_ms={:.3f} import_images_ms={:.3f} "
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
      metrics.file_read_ms.total_ms,
      metrics.file_read_ms.AverageMs(),
      metrics.file_read_ms.max_ms,
      metrics.image_parse_ms.total_ms,
      metrics.image_parse_ms.AverageMs(),
      metrics.image_parse_ms.max_ms,
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

  // Per-stage benchmark table
  const auto pct = [&](const TimerStats& s) -> double {
    return metrics.wall_total_ms > 0.0
               ? 100.0 * s.total_ms / metrics.wall_total_ms
               : 0.0;
  };
  const double wall_avg_ms =
      metrics.num_extracted > 0
          ? metrics.wall_total_ms / metrics.num_extracted
          : 0.0;
  spdlog::info(
      "Phoenix extraction benchmark:"
      " extracted={} skipped={} wall={:.0f}ms {:.2f} img/s",
      metrics.num_extracted,
      metrics.num_skipped,
      metrics.wall_total_ms,
      images_per_sec);
  spdlog::info("  {:<26} {:>10} {:>8} {:>8} {:>6}",
               "Stage", "Total(ms)", "Avg(ms)", "Max(ms)", "%wall");
  spdlog::info("  {}", std::string(60, '-'));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f} {:>8.1f} {:>5.1f}%",
               "[2] file_read (parallel)",
               metrics.file_read_ms.total_ms,
               metrics.file_read_ms.AverageMs(),
               metrics.file_read_ms.max_ms,
               pct(metrics.file_read_ms));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f} {:>8.1f} {:>5.1f}%",
               "[3] img_decode (parallel)",
               metrics.image_parse_ms.total_ms,
               metrics.image_parse_ms.AverageMs(),
               metrics.image_parse_ms.max_ms,
               pct(metrics.image_parse_ms));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f} {:>8.1f} {:>5.1f}%",
               "[4a] detect (ALIKED)",
               metrics.detect_ms.total_ms,
               metrics.detect_ms.AverageMs(),
               metrics.detect_ms.max_ms,
               pct(metrics.detect_ms));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f} {:>8.1f} {:>5.1f}%",
               "[4b] convert",
               metrics.convert_ms.total_ms,
               metrics.convert_ms.AverageMs(),
               metrics.convert_ms.max_ms,
               pct(metrics.convert_ms));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f} {:>8.1f} {:>5.1f}%",
               "[4c] db_write",
               metrics.db_write_ms.total_ms,
               metrics.db_write_ms.AverageMs(),
               metrics.db_write_ms.max_ms,
               pct(metrics.db_write_ms));
  spdlog::info("  {}", std::string(60, '-'));
  spdlog::info("  {:<26} {:>10.1f} {:>8.1f}",
               "wall_total",
               metrics.wall_total_ms,
               wall_avg_ms);
}

// Core extraction logic — shared by ExecFeatureExtractor and RunFeatureExtractor.
int DoExtraction(const std::string& database_path,
                 colmap::ImageReaderOptions reader_options,
                 const FeatureExtractionOptions& cli_options) {
  const std::filesystem::path image_path = reader_options.image_path;
  WriteExtractionMaxEdgeMetadata(database_path, cli_options.max_edge);

  FeatureExtractionMetrics metrics;
  const auto extraction_start = Clock::now();

  SelectImagesForExtraction(
      cli_options, image_path, &reader_options, &metrics);
  if (ShouldExitEarlyAfterSelection(cli_options, reader_options)) {
    metrics.wall_total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  extraction_start)
            .count();
    LogFeatureExtractionMetrics(metrics);
    return EXIT_SUCCESS;
  }

  reader_options.Check();

  auto database = colmap::Database::Open(database_path);
  const bool default_share_camera_per_folder =
      DefaultShareCameraPerFolder(cli_options, reader_options);

  const int detect_batch_size = kPhoenixFixedBatchSize;

  sfm_phoenix::AlikedDetector detector;
  if (!InitializeDetector(cli_options, detect_batch_size, &metrics, &detector)) {
    return EXIT_FAILURE;
  }
  spdlog::info("Using fixed detect_batch_size={}", detect_batch_size);

  std::vector<std::string> existing_image_names;
  existing_image_names.reserve(reader_options.image_names.size());
  std::unordered_map<std::string, ImportedCameraEntry> camera_cache;
  ExtractorExecutionContext execution_context = {
      reader_options,
      default_share_camera_per_folder,
      cli_options.skip_existing,
      detect_batch_size,
      database.get(),
      &detector,
      &metrics,
      &camera_cache,
  };

  ProcessNewImages(reader_options.image_names,
                   reader_options,
                   &execution_context,
                   &existing_image_names);

  if (IsInterruptRequested()) {
    metrics.wall_total_ms =
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  extraction_start)
            .count();
    spdlog::warn("Phoenix feature extraction interrupted by Ctrl+C");
    LogFeatureExtractionMetrics(metrics);
    return EXIT_FAILURE;
  }

  if (!existing_image_names.empty()) {
    const std::vector<colmap::Image> existing_images =
        CollectTargetImages(*database, existing_image_names);
    if (!ProcessExistingImages(existing_images,
                               reader_options.image_path,
                               cli_options.skip_existing,
                               database.get(),
                               &detector,
                               &metrics)) {
      if (IsInterruptRequested()) {
        metrics.wall_total_ms =
            std::chrono::duration<double, std::milli>(Clock::now() -
                                                      extraction_start)
                .count();
        spdlog::warn("Phoenix feature extraction interrupted by Ctrl+C");
        LogFeatureExtractionMetrics(metrics);
      }
      return EXIT_FAILURE;
    }
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

}  // namespace

int ExecFeatureExtractor(const std::string& database_path,
                         const std::filesystem::path& image_path,
                         const FeatureExtractionOptions& opts) {
  FeatureExtractionOptions clamped = opts;
  ClampFeatureExtractionOptions(&clamped);
  spdlog::info("Phoenix.max_edge={} top_k={} scores_th={:.3f}",
               clamped.max_edge, clamped.top_k, clamped.scores_th);
  return DoExtraction(
      database_path, BuildReaderOptionsFromPath(image_path, clamped), clamped);
}

int RunFeatureExtractor(int argc, char** argv) {
  FeatureExtractionOptions cli_options;
  colmap::OptionManager options;
  AddFeatureExtractorOptions(&options, &cli_options);
  options.Parse(argc, argv);
  ClampFeatureExtractionOptions(&cli_options);
  spdlog::info("Phoenix.max_edge={} top_k={} scores_th={:.3f}",
               cli_options.max_edge,
               cli_options.top_k,
               cli_options.scores_th);
  return DoExtraction(*options.database_path,
                      BuildReaderOptions(options, cli_options),
                      cli_options);
}

}  // namespace phoenix_tool