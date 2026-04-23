#pragma once

#include <colmap/controllers/image_reader.h>
#include <colmap/scene/database.h>

#include <opencv2/core/mat.hpp>

#include <Eigen/Core>

#include <algorithm>
#include <filesystem>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace phoenix_tool {

struct TimerStats {
  double total_ms = 0.0;
  double max_ms = 0.0;
  int samples = 0;

  void AddSample(const double value_ms) {
    total_ms += value_ms;
    max_ms = std::max(max_ms, value_ms);
    ++samples;
  }

  double AverageMs() const {
    return samples > 0 ? total_ms / samples : 0.0;
  }
};

struct CachedFeatures {
  colmap::FeatureKeypoints colmap_keypoints;
  std::vector<Eigen::Vector2d> points;
  std::vector<float> keypoints;
  std::vector<float> descriptors;
  int num_keypoints = 0;
};

struct FeatureExtractionMetrics {
  int candidate_images = 0;
  int selected_images = 0;
  int num_extracted = 0;
  int num_skipped = 0;
  int num_filtered_static_frames = 0;
  TimerStats static_filter_ms;
  TimerStats file_read_ms;
  TimerStats image_parse_ms;
  TimerStats import_images_ms;
  TimerStats image_registration_ms;
  TimerStats feature_check_ms;
  TimerStats engine_init_ms;
  TimerStats image_decode_ms;
  TimerStats detect_ms;
  TimerStats convert_ms;
  TimerStats db_write_ms;
  double wall_total_ms = 0.0;
};

struct FeatureMatchingMetrics {
  int scheduled_pairs = 0;
  int num_matched_pairs = 0;
  int num_skipped_pairs = 0;
  int num_zero_match_pairs = 0;
  int feature_cache_hits = 0;
  int feature_cache_misses = 0;
  int cached_images = 0;
  TimerStats pair_prepare_ms;
  TimerStats retrieval_ms;
  TimerStats engine_init_ms;
  TimerStats feature_load_ms;
  TimerStats feature_db_read_ms;
  TimerStats feature_convert_ms;
  TimerStats match_ms;
  TimerStats geometry_ms;
  TimerStats db_write_ms;
  double wall_total_ms = 0.0;
};

// Options for DINOv3-based image retrieval pair generation.
struct RetrievalOptions {
  int num = 50;
  int batch_size = 32;
  double similarity_threshold = 0.0;
  double relative_threshold = 0.8;
};

// Options for ALIKED feature extraction (phoenix feature_extractor).
struct FeatureExtractionOptions {
  int camera_mode = -1;
  int max_edge = 1024;
  int top_k = 5000;
  double scores_th = 0.2;
  int detect_batch_size = 0;  // 0 = auto (GPU memory based)
  double static_frame_diff_threshold = 1.0;
  bool single_camera_per_folder = true;
  bool filter_static_frames = false;
  bool skip_existing = true;
};

// Options for LightGlue feature matching (phoenix feature_matcher).
struct FeatureMatchingOptions {
  enum class Preset {
    kFeatureMatcher,
    kAutomaticReconstructor,
  };

  explicit FeatureMatchingOptions(
      const Preset preset = Preset::kFeatureMatcher)
      : linear_overlap_num(
            preset == Preset::kAutomaticReconstructor ? 0 : 15),
        quadratic_overlap_num(
            preset == Preset::kAutomaticReconstructor ? 0 : 10) {}

  std::filesystem::path image_path;   // image root dir; used for retrieval
  int max_matches = 4000;
  int linear_overlap_num;
  int quadratic_overlap_num;
  bool skip_existing = true;
  bool overwrite_existing = false;
  RetrievalOptions retrieval;
};

// Options for the full automatic reconstruction pipeline.
struct AutomaticOptions {
  std::string database_path;
  std::string image_path;
  std::string output_path;
  FeatureExtractionOptions extraction;
  FeatureMatchingOptions matching =
      FeatureMatchingOptions(
          FeatureMatchingOptions::Preset::kAutomaticReconstructor);
  std::vector<std::string> mapper_args;
};

void ApplyCameraMode(int camera_mode,
                     colmap::ImageReaderOptions* reader_options);

std::vector<Eigen::Vector2d> ToPoints(
    const colmap::FeatureKeypoints& keypoints);

int RunCommand(const std::string& executable,
               const std::vector<std::string>& args,
               const std::string& prefix = "");

std::filesystem::path ExecutableDirectory(const char* argv0);

void PrintHelp();

std::vector<std::string> SliceArgs(int argc, char** argv, int start);

std::vector<std::string> ReadTextLines(
    const std::filesystem::path& path,
    bool allow_comments = false);

std::vector<std::string> CollectImageNamesFromDirectory(
  const std::filesystem::path& image_path,
  bool recursive);

std::vector<std::string> FilterStaticAdjacentFrames(
    const std::filesystem::path& image_path,
    const std::vector<std::string>& image_names,
    double static_frame_diff_threshold,
    int* num_filtered_static_frames);

void ImportImages(const colmap::ImageReaderOptions& reader_options,
                  colmap::Database* database);

std::vector<colmap::Image> CollectTargetImages(
    const colmap::Database& database,
    const std::vector<std::string>& image_names);

std::vector<colmap::Image> LoadImages(const colmap::Database& database);

std::vector<colmap::Image> SelectOrderedImages(
    const std::vector<colmap::Image>& all_images,
    const std::vector<std::string>& image_names);

std::vector<std::pair<colmap::image_t, colmap::image_t>> BuildExhaustivePairs(
    const std::vector<colmap::Image>& images);

std::vector<std::pair<colmap::image_t, colmap::image_t>> BuildSequentialPairs(
    const std::vector<colmap::Image>& images,
    int linear_overlap_num,
    int quadratic_overlap_num);

const CachedFeatures& LoadFeatures(
    const colmap::Database& database,
    const std::unordered_map<colmap::image_t, colmap::Image>& images,
    const std::unordered_map<colmap::camera_t, colmap::Camera>& cameras,
    int max_edge,
    colmap::image_t image_id,
    std::unordered_map<colmap::image_t, CachedFeatures>* cache,
    FeatureMatchingMetrics* metrics);

bool ParseBool(const std::string& value);

void WriteExtractionMaxEdgeMetadata(const std::string& database_path,
                                    int max_edge);

bool ReadExtractionMaxEdgeMetadata(const std::string& database_path,
                                   int* max_edge);

void InstallInterruptHandler();
void ResetInterruptRequested();
bool IsInterruptRequested();

AutomaticOptions ParseAutomaticOptions(int argc, char** argv);

// Direct execution (no CLI arg parsing).
int ExecFeatureExtractor(const std::string& database_path,
                         const std::filesystem::path& image_path,
                         const FeatureExtractionOptions& opts);
int ExecFeatureMatcher(const std::string& database_path,
                       const FeatureMatchingOptions& opts);

// CLI entry points (parse argc/argv then delegate to Exec* above).
int RunFeatureExtractor(int argc, char** argv);
int RunFeatureMatcher(int argc, char** argv);
int RunAutomaticReconstructor(const AutomaticOptions& options);

}  // namespace phoenix_tool