#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/extractors/aliked.h"
#include "sfm_phoenix/internal/model_runtime.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/sensor/rig.h>

#include <oneapi/tbb/parallel_pipeline.h>
#include <opencv2/imgcodecs.hpp>
#include <spdlog/spdlog.h>

#include <chrono>
#include <fstream>
#include <thread>
#include <unordered_map>
#include <vector>

namespace phoenix_tool {

namespace {

using Clock = std::chrono::steady_clock;

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

struct FeatureExtractorCliOptions {
  std::filesystem::path image_list_path;
  int camera_mode = -1;
  int max_edge = 1600;
  int top_k = 5000;
  double scores_th = 0.2;
  double static_frame_diff_threshold = 1.0;
  bool filter_static_frames = false;
  bool skip_existing = true;
};

struct ExtractorExecutionContext {
  const colmap::ImageReaderOptions& reader_options;
  bool default_share_camera_per_folder = false;
  bool skip_existing = true;
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
                                FeatureExtractorCliOptions* cli_options) {
  options->AddDatabaseOptions();
  options->AddImageOptions();
  options->AddDefaultOption("camera_mode", &cli_options->camera_mode);
  options->AddDefaultOption("image_list_path", &cli_options->image_list_path);
  options->AddFeatureExtractionOptions();
  options->AddDefaultOption("Phoenix.max_edge", &cli_options->max_edge);
  options->AddDefaultOption("Phoenix.top_k", &cli_options->top_k);
  options->AddDefaultOption("Phoenix.scores_th", &cli_options->scores_th);
  options->AddDefaultOption("Phoenix.filter_static_frames",
                            &cli_options->filter_static_frames);
  options->AddDefaultOption("Phoenix.static_frame_diff_threshold",
                            &cli_options->static_frame_diff_threshold);
  options->AddDefaultOption("Phoenix.skip_existing",
                            &cli_options->skip_existing);
}

void ClampFeatureExtractorOptions(FeatureExtractorCliOptions* cli_options) {
  constexpr int kMaxExtractedFeatures = 5000;
  if (cli_options->top_k > kMaxExtractedFeatures) {
    spdlog::warn("Phoenix.top_k={} exceeds hard cap {}, clamping",
                 cli_options->top_k,
                 kMaxExtractedFeatures);
    cli_options->top_k = kMaxExtractedFeatures;
  }
}

colmap::ImageReaderOptions BuildReaderOptions(
    const colmap::OptionManager& options,
    const FeatureExtractorCliOptions& cli_options) {
  colmap::ImageReaderOptions reader_options = *options.image_reader;
  reader_options.image_path = *options.image_path;
  reader_options.as_rgb = true;

  if (cli_options.camera_mode >= 0) {
    ApplyCameraMode(cli_options.camera_mode, &reader_options);
  }

  if (!cli_options.image_list_path.empty()) {
    reader_options.image_names = ReadTextLines(cli_options.image_list_path);
  }

  return reader_options;
}

void SelectImagesForExtraction(
    const FeatureExtractorCliOptions& cli_options,
    const std::filesystem::path& image_path,
    colmap::ImageReaderOptions* reader_options,
    FeatureExtractionMetrics* metrics) {
  std::vector<std::string> candidate_image_names = reader_options->image_names;
  if (candidate_image_names.empty()) {
    candidate_image_names = CollectImageNamesFromDirectory(image_path);
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
    const FeatureExtractorCliOptions& cli_options,
    const colmap::ImageReaderOptions& reader_options) {
  return (cli_options.filter_static_frames ||
          !cli_options.image_list_path.empty()) &&
         reader_options.image_names.empty();
}

bool DefaultShareCameraPerFolder(
    const FeatureExtractorCliOptions& cli_options,
    const colmap::ImageReaderOptions& reader_options) {
  return cli_options.camera_mode < 0 &&
         reader_options.existing_camera_id == colmap::kInvalidCameraId &&
         !reader_options.single_camera &&
         !reader_options.single_camera_per_folder &&
         !reader_options.single_camera_per_image;
}

bool InitializeDetector(const FeatureExtractorCliOptions& cli_options,
                       FeatureExtractionMetrics* metrics,
                       sfm_phoenix::AlikedDetector* detector) {
  constexpr int kMaxExtractedFeatures = 5000;
  const auto engine_init_start = Clock::now();
  const auto backbone_engine = sfm_phoenix::EnsurePhoenixBackboneEngine();
  const auto sddh_engine = sfm_phoenix::EnsurePhoenixSddhEngine();

  sfm_phoenix::AlikedConfig detector_config;
  detector_config.backbone_engine = backbone_engine.string();
  detector_config.sddh_engine = sddh_engine.string();
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

void ProcessNewImageTask(const DecodedImageTask& task,
                        ExtractorExecutionContext* context) {
  context->metrics->file_read_ms.AddSample(task.file_read_ms);
  context->metrics->image_parse_ms.AddSample(task.image_parse_ms);
  context->metrics->image_decode_ms.AddSample(task.decode_ms);

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

  const colmap::image_t image_id = RegisterImageAndFrame(task.image_name,
                                                         camera_entry,
                                                         context->database,
                                                         context->metrics);

  const FeatureAvailability availability = CheckFeatureAvailability(
      context->database,
      image_id,
      task.image_name,
      context->skip_existing,
      context->metrics);
  if (availability == FeatureAvailability::kSkip) {
    return;
  }
  if (availability == FeatureAvailability::kConflict) {
    throw std::runtime_error("Image already has features in database: " +
                             task.image_name);
  }

  DetectAndStoreFeatures(task.image_name,
                         task.image_bgr,
                         image_id,
                         context->detector,
                         context->database,
                         context->metrics);
}

void ProcessNewImages(
    const std::vector<std::string>& image_names,
    const colmap::ImageReaderOptions& reader_options,
    ExtractorExecutionContext* context,
    std::vector<std::string>* existing_image_names) {
  size_t next_image_index = 0;
  const size_t decode_tokens = DecodePipelineTokens();

  context->database->BeginTransaction();
  tbb::parallel_pipeline(
      decode_tokens,
      tbb::make_filter<void, DecodedImageTask>(
          tbb::filter_mode::serial_in_order,
          [&](tbb::flow_control& control) -> DecodedImageTask {
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
                ProcessNewImageTask(task, context);
              }));
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
  for (const auto& image : existing_images) {
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

}  // namespace

int RunFeatureExtractor(int argc, char** argv) {
  FeatureExtractorCliOptions cli_options;

  colmap::OptionManager options;
  AddFeatureExtractorOptions(&options, &cli_options);
  options.Parse(argc, argv);
  ClampFeatureExtractorOptions(&cli_options);

  colmap::ImageReaderOptions reader_options =
      BuildReaderOptions(options, cli_options);

  FeatureExtractionMetrics metrics;
  const auto extraction_start = Clock::now();

  SelectImagesForExtraction(
      cli_options, *options.image_path, &reader_options, &metrics);
  if (ShouldExitEarlyAfterSelection(cli_options, reader_options)) {
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
      DefaultShareCameraPerFolder(cli_options, reader_options);

  sfm_phoenix::AlikedDetector detector;
  if (!InitializeDetector(cli_options, &metrics, &detector)) {
    return EXIT_FAILURE;
  }

  std::vector<std::string> existing_image_names;
  existing_image_names.reserve(reader_options.image_names.size());
  std::unordered_map<std::string, ImportedCameraEntry> camera_cache;
  ExtractorExecutionContext execution_context = {
      reader_options,
      default_share_camera_per_folder,
      cli_options.skip_existing,
      database.get(),
      &detector,
      &metrics,
      &camera_cache,
  };

  ProcessNewImages(reader_options.image_names,
                   reader_options,
                   &execution_context,
                   &existing_image_names);

  if (!existing_image_names.empty()) {
    const std::vector<colmap::Image> existing_images =
        CollectTargetImages(*database, existing_image_names);
    if (!ProcessExistingImages(existing_images,
                               reader_options.image_path,
                               cli_options.skip_existing,
                               database.get(),
                               &detector,
                               &metrics)) {
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

}  // namespace phoenix_tool