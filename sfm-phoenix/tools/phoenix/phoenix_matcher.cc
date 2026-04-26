#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/internal/model_runtime.h"
#include "sfm_phoenix/internal/trt_engine.h"
#include "sfm_phoenix/matchers/lightglue.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/estimators/two_view_geometry.h>

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>

#include <Eigen/Core>

#include <cuda_runtime.h>

#include <spdlog/spdlog.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <deque>
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <unordered_set>

namespace phoenix_tool {

namespace {

constexpr int kPhoenixFixedBatchSize = 4;
constexpr int kPhoenixMaxBatchSize = 8;
constexpr int kPhoenixMaxKeypoints = 5000;

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

size_t EstimateLightGlueEngineReserveBytes(const int batch_size) {
  switch (ClampSupportedPhoenixBatchSize(batch_size, /*max_supported=*/8)) {
    case 8:
      return static_cast<size_t>(3200) * 1024 * 1024;
    case 4:
      return static_cast<size_t>(1800) * 1024 * 1024;
    default:
      return static_cast<size_t>(700) * 1024 * 1024;
  }
}

int EstimateMatchingBatchSize(const size_t per_pair_staging,
                              const int max_supported) {
  size_t free_vram_bytes = 0, total_vram_bytes = 0;
  if (cudaMemGetInfo(&free_vram_bytes, &total_vram_bytes) != cudaSuccess) {
    spdlog::warn("cudaMemGetInfo failed, defaulting matching batch_size=1");
    return 1;
  }

  const size_t cuda_safety_bytes = static_cast<size_t>(256) * 1024 * 1024;
  for (const int candidate : {8, 4, 1}) {
    if (candidate > max_supported) {
      continue;
    }
    const size_t required_bytes = EstimateLightGlueEngineReserveBytes(candidate) +
                                  per_pair_staging * candidate +
                                  cuda_safety_bytes;
    if (free_vram_bytes >= required_bytes) {
      return candidate;
    }
  }
  return 1;
}

uint64_t PairKey(const colmap::image_t image_id1,
                 const colmap::image_t image_id2) {
  const auto min_id = static_cast<uint32_t>(std::min(image_id1, image_id2));
  const auto max_id = static_cast<uint32_t>(std::max(image_id1, image_id2));
  return (static_cast<uint64_t>(min_id) << 32) | max_id;
}

std::filesystem::path DefaultRetrievalModelPath() {
  return std::filesystem::current_path() / "sfm-phoenix" / "models" /
         "dinov3_vitb16_pretrain_lvd1689m.onnx";
}

void MergeUniquePairs(
    const std::vector<std::pair<colmap::image_t, colmap::image_t>>& extra_pairs,
    std::vector<std::pair<colmap::image_t, colmap::image_t>>* pairs) {
  std::unordered_set<uint64_t> seen_pairs;
  seen_pairs.reserve(pairs->size() + extra_pairs.size());
  for (const auto& [image_id1, image_id2] : *pairs) {
    seen_pairs.insert(PairKey(image_id1, image_id2));
  }
  for (const auto& [image_id1, image_id2] : extra_pairs) {
    if (image_id1 == image_id2) {
      continue;
    }
    if (seen_pairs.insert(PairKey(image_id1, image_id2)).second) {
      pairs->emplace_back(image_id1, image_id2);
    }
  }
}

// --- DINOv3 TRT Retrieval ---

static constexpr int kDinoInputSize = 224;
static constexpr int kDinoEmbedDim = 768;
static constexpr float kDinoMean[3] = {0.485f, 0.456f, 0.406f};
static constexpr float kDinoStd[3] = {0.229f, 0.224f, 0.225f};

// Preprocess one BGR image into the CHW float buffer expected by DINOv3.
// Resizes to 224×224, converts BGR→RGB, normalises with ImageNet stats.
void PreprocessDino(const cv::Mat& bgr, float* out_chw) {
  cv::Mat resized;
  cv::resize(bgr, resized, cv::Size(kDinoInputSize, kDinoInputSize),
             0, 0, cv::INTER_AREA);
  std::vector<cv::Mat> chans(3);
  cv::split(resized, chans);  // chans[0]=B, [1]=G, [2]=R (OpenCV BGR order)
  // Map output channel c to BGR index: R→chans[2], G→chans[1], B→chans[0]
  const int bgr_idx[3] = {2, 1, 0};
  for (int c = 0; c < 3; ++c) {
    cv::Mat fc;
    chans[bgr_idx[c]].convertTo(
        fc, CV_32F,
        1.0f / (255.0f * kDinoStd[c]),
        -kDinoMean[c] / kDinoStd[c]);
    std::memcpy(out_chw + c * kDinoInputSize * kDinoInputSize,
                fc.ptr<float>(),
                kDinoInputSize * kDinoInputSize * sizeof(float));
  }
}

// Run batched DINOv3 inference. Returns L2-normalised embeddings [N × 768].
std::vector<std::vector<float>> ExtractDinoEmbeddings(
    const std::vector<std::string>& image_names,
    const std::filesystem::path& image_dir,
    sfm_phoenix::TrtEngine& engine,
    const int batch_size) {
  const int n = static_cast<int>(image_names.size());
  const int pixels = 3 * kDinoInputSize * kDinoInputSize;

  cudaStream_t stream = nullptr;
  cudaStreamCreate(&stream);

  std::vector<float> batch_input(static_cast<size_t>(batch_size) * pixels);
  std::vector<float> batch_output(
      static_cast<size_t>(batch_size) * kDinoEmbedDim);
  std::vector<std::vector<float>> result(n);
  ResourceSnapshot previous_snapshot = CaptureResourceSnapshot();

  for (int i = 0; i < n; i += batch_size) {
    const int actual = std::min(batch_size, n - i);
    for (int j = 0; j < actual; ++j) {
      const auto path = image_dir / image_names[i + j];
      cv::Mat bgr = cv::imread(path.string(), cv::IMREAD_COLOR);
      if (bgr.empty()) {
        spdlog::warn("Phoenix retrieval: failed to load {}", path.string());
        std::fill_n(batch_input.data() + j * pixels, pixels, 0.0f);
      } else {
        PreprocessDino(bgr, batch_input.data() + j * pixels);
      }
    }

    engine.set_input_shape("pixel_values",
                           {actual, 3, kDinoInputSize, kDinoInputSize});
    engine.set_input("pixel_values",
                     batch_input.data(),
                     static_cast<size_t>(actual) * pixels * sizeof(float),
                     stream);
    engine.infer(stream);
    engine.get_output(
        "embeddings",
        batch_output.data(),
        static_cast<size_t>(actual) * kDinoEmbedDim * sizeof(float),
        stream);
    cudaStreamSynchronize(stream);
    const ResourceSnapshot current_snapshot = CaptureResourceSnapshot();

    for (int j = 0; j < actual; ++j) {
      float* emb = batch_output.data() + j * kDinoEmbedDim;
      float sq_sum = 0.0f;
      for (int k = 0; k < kDinoEmbedDim; ++k) sq_sum += emb[k] * emb[k];
      const float inv_norm = 1.0f / (std::sqrt(sq_sum) + 1e-12f);
      std::vector<float> v(kDinoEmbedDim);
      for (int k = 0; k < kDinoEmbedDim; ++k) v[k] = emb[k] * inv_norm;
      result[i + j] = std::move(v);
    }
    spdlog::info("Phoenix retrieval: embeddings {}/{}", i + actual, n);
    LogResourceSnapshot("retrieval.batch.done",
                        current_snapshot,
                        &previous_snapshot,
                        "images=" + std::to_string(actual) +
                            " upto=" + std::to_string(i + actual) +
                            "/" + std::to_string(n));
    previous_snapshot = current_snapshot;
  }

  cudaStreamDestroy(stream);
  return result;
}

// Build top-K pairs per image from L2-normalised embeddings via brute-force
// dot-product (= cosine similarity). Returns unique (min_id, max_id) pairs.
std::vector<std::pair<colmap::image_t, colmap::image_t>> TopKRetrievalPairs(
    const std::vector<std::string>& image_names,
    const std::vector<std::vector<float>>& embeddings,
    const int top_k,
    const int query_period,
  const double similarity_threshold,
  const double relative_threshold,
    const std::unordered_map<std::string, colmap::image_t>& image_ids) {
  const int n = static_cast<int>(embeddings.size());

  Eigen::MatrixXf E(n, kDinoEmbedDim);
  for (int i = 0; i < n; ++i) {
    E.row(i) =
        Eigen::Map<const Eigen::VectorXf>(embeddings[i].data(), kDinoEmbedDim);
  }
  const Eigen::MatrixXf S = E * E.transpose();  // cosine similarity [n × n]

  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  pairs.reserve(static_cast<size_t>(n) * top_k);

  std::vector<std::pair<float, int>> row_scores;
  row_scores.reserve(n);

  for (int i = 0; i < n; ++i) {
    if (query_period > 1 && ((i + 1) % query_period) != 0 && i + 1 != n) {
      continue;
    }
    const auto it_i = image_ids.find(image_names[i]);
    if (it_i == image_ids.end()) continue;

    row_scores.clear();
    for (int j = 0; j < n; ++j) {
      if (i == j) continue;
      if (image_ids.count(image_names[j]) == 0) continue;
      row_scores.emplace_back(S(i, j), j);
    }

    const int k = std::min(top_k, static_cast<int>(row_scores.size()));
    std::partial_sort(row_scores.begin(), row_scores.begin() + k,
                      row_scores.end(),
                      std::greater<std::pair<float, int>>());
    const float best_score = k > 0 ? row_scores.front().first : -1.0f;

    for (int ki = 0; ki < k; ++ki) {
      const float score = row_scores[ki].first;
      if (similarity_threshold > 0.0 && score < similarity_threshold) {
        continue;
      }
      if (relative_threshold > 0.0 &&
          score < best_score * relative_threshold) {
        continue;
      }
      const auto it_j =
          image_ids.find(image_names[row_scores[ki].second]);
      const colmap::image_t id_i = it_i->second;
      const colmap::image_t id_j = it_j->second;
      pairs.emplace_back(std::min(id_i, id_j), std::max(id_i, id_j));
    }
  }

  return pairs;
}

std::vector<std::pair<colmap::image_t, colmap::image_t>> BuildRetrievalPairs(
    const std::filesystem::path& image_path,
    const int retrieval_num,
    const int retrieval_batch_size,
  const int retrieval_period,
  const double similarity_threshold,
  const double relative_threshold,
    const std::unordered_map<std::string, colmap::image_t>& image_ids) {
  if (retrieval_num <= 0) return {};
  if (image_path.empty()) {
    throw std::runtime_error(
        "Phoenix retrieval requires --image_path when "
        "Phoenix.retrieval_num > 0");
  }

  const auto onnx_path = DefaultRetrievalModelPath();

  const auto t_engine_start = std::chrono::steady_clock::now();
  const ResourceSnapshot before_engine = CaptureResourceSnapshot();
  LogResourceSnapshot("retrieval.engine.before_init", before_engine);
  const auto engine_path = sfm_phoenix::EnsureRetrievalEngine(onnx_path);

  spdlog::info(
      "Phoenix retrieval: generating top-{} pairs via TRT engine {} "
      "similarity_threshold={:.3f} relative_threshold={:.3f}",
      retrieval_num,
      engine_path.string(),
      similarity_threshold,
      relative_threshold);

  sfm_phoenix::TrtEngine engine;
  if (!engine.load(engine_path.string())) {
    throw std::runtime_error("Failed to load retrieval engine: " +
                             engine_path.string());
  }
  const ResourceSnapshot after_engine = CaptureResourceSnapshot();
  const double engine_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_engine_start)
          .count();
  LogResourceSnapshot("retrieval.engine.after_init",
                      after_engine,
                      &before_engine,
                      "engine=" + engine_path.filename().string() +
                          " engine_ms=" + std::to_string(engine_ms));

  // Collect image names — ordered or from DB
  std::vector<std::string> image_names;
  for (const auto& [name, id] : image_ids) image_names.push_back(name);
  std::sort(image_names.begin(), image_names.end());
  if (image_names.empty()) return {};

  const auto t_embed_start = std::chrono::steady_clock::now();
  const auto embeddings = ExtractDinoEmbeddings(
      image_names, image_path, engine, retrieval_batch_size);
  const double embed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_embed_start)
          .count();

  const auto t_topk_start = std::chrono::steady_clock::now();
  auto pairs = TopKRetrievalPairs(image_names,
                                  embeddings,
                                  retrieval_num,
                                  retrieval_period,
                                  similarity_threshold,
                                  relative_threshold,
                                  image_ids);
  const double topk_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_topk_start)
          .count();

  spdlog::info(
      "Phoenix retrieval: {} pairs generated | "
      "engine_ms={:.1f} embed_ms={:.1f} topk_ms={:.1f} total_ms={:.1f}",
      pairs.size(), engine_ms, embed_ms, topk_ms,
      engine_ms + embed_ms + topk_ms);
  return pairs;
}

void LogFeatureMatchingMetrics(const FeatureMatchingMetrics& metrics) {
  const double pairs_per_sec =
      metrics.wall_total_ms > 0.0
          ? (1000.0 * metrics.num_matched_pairs) / metrics.wall_total_ms
          : 0.0;

  spdlog::info(
      "Phoenix matching performance: scheduled_pairs={} matched_pairs={} "
      "skipped_pairs={} zero_match_pairs={} cached_images={} "
      "feature_cache_hits={} feature_cache_misses={} pair_prepare_ms={:.3f} "
      "retrieval_total_ms={:.3f} "
      "engine_init_ms={:.3f} feature_load_total_ms={:.3f} "
      "feature_load_avg_ms={:.3f} feature_load_max_ms={:.3f} "
      "feature_db_read_total_ms={:.3f} feature_db_read_avg_ms={:.3f} "
      "feature_convert_total_ms={:.3f} feature_convert_avg_ms={:.3f} "
      "match_total_ms={:.3f} match_avg_ms={:.3f} match_max_ms={:.3f} "
      "geometry_total_ms={:.3f} geometry_avg_ms={:.3f} geometry_max_ms={:.3f} "
      "db_write_total_ms={:.3f} db_write_avg_ms={:.3f} db_write_max_ms={:.3f} "
      "wall_total_ms={:.3f} wall_avg_ms={:.3f} pairs_per_sec={:.3f}",
      metrics.scheduled_pairs,
      metrics.num_matched_pairs,
      metrics.num_skipped_pairs,
      metrics.num_zero_match_pairs,
      metrics.cached_images,
      metrics.feature_cache_hits,
      metrics.feature_cache_misses,
      metrics.pair_prepare_ms.total_ms,
      metrics.retrieval_ms.total_ms,
      metrics.engine_init_ms.total_ms,
      metrics.feature_load_ms.total_ms,
      metrics.feature_load_ms.AverageMs(),
      metrics.feature_load_ms.max_ms,
      metrics.feature_db_read_ms.total_ms,
      metrics.feature_db_read_ms.AverageMs(),
      metrics.feature_convert_ms.total_ms,
      metrics.feature_convert_ms.AverageMs(),
      metrics.match_ms.total_ms,
      metrics.match_ms.AverageMs(),
      metrics.match_ms.max_ms,
      metrics.geometry_ms.total_ms,
      metrics.geometry_ms.AverageMs(),
      metrics.geometry_ms.max_ms,
      metrics.db_write_ms.total_ms,
      metrics.db_write_ms.AverageMs(),
      metrics.db_write_ms.max_ms,
      metrics.wall_total_ms,
      metrics.num_matched_pairs > 0
          ? metrics.wall_total_ms / metrics.num_matched_pairs
          : 0.0,
      pairs_per_sec);
}

// Registers FeatureMatchingOptions fields with the COLMAP OptionManager.
void AddFeatureMatcherOptions(colmap::OptionManager* options,
                               FeatureMatchingOptions* opts) {
  options->AddDatabaseOptions();
  options->AddDefaultOption("image_path", &opts->image_path);
  options->AddDefaultOption("Phoenix.max_matches", &opts->max_matches);
  options->AddDefaultOption("Phoenix.retrieval_num", &opts->retrieval.num);
  options->AddDefaultOption("Phoenix.retrieval_batch_size",
                            &opts->retrieval.batch_size);
  options->AddDefaultOption("SequentialMatching.loop_detection_period",
                            &opts->retrieval.period);
  options->AddDefaultOption("SequentialMatching.loop_detection_num_images",
                            &opts->retrieval.num);
  options->AddDefaultOption("Phoenix.retrieval_similarity_threshold",
                            &opts->retrieval.similarity_threshold);
  options->AddDefaultOption("Phoenix.retrieval_relative_threshold",
                            &opts->retrieval.relative_threshold);
  options->AddDefaultOption("Phoenix.linear_overlap_num",
                            &opts->linear_overlap_num);
  options->AddDefaultOption("Phoenix.quadratic_overlap_num",
                            &opts->quadratic_overlap_num);
  options->AddDefaultOption("Phoenix.skip_existing", &opts->skip_existing);
  options->AddDefaultOption("Phoenix.overwrite_existing",
                            &opts->overwrite_existing);
}

// Core matching logic — shared by ExecFeatureMatcher and RunFeatureMatcher.
int DoMatching(const std::string& database_path,
               const FeatureMatchingOptions& opts) {
  constexpr int kMaxMatchedFeatures = 4000;
  using Clock = std::chrono::steady_clock;

  int extraction_max_edge = 0;
  if (!ReadExtractionMaxEdgeMetadata(database_path, &extraction_max_edge)) {
    throw std::runtime_error(
        "Phoenix extraction metadata missing: extraction_max_edge not found in "
        "database. Run feature_extractor again with this Phoenix version.");
  }

  FeatureMatchingMetrics metrics;
  const auto matching_start = Clock::now();

  auto database = colmap::Database::Open(database_path);
  const std::vector<colmap::Image> all_images = LoadImages(*database);
  const std::vector<colmap::Image> ordered_images =
      SelectOrderedImages(all_images, {});

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

  const auto pair_prepare_start = Clock::now();
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  if (opts.linear_overlap_num > 0 || opts.quadratic_overlap_num > 0) {
    pairs = BuildSequentialPairs(ordered_images,
                                 opts.linear_overlap_num,
                                 opts.quadratic_overlap_num);
  } else if (opts.retrieval.num <= 0) {
    pairs = BuildExhaustivePairs(ordered_images);
  }
  if (opts.retrieval.num > 0) {
    const auto retrieval_start = Clock::now();
    const auto retrieval_pairs =
        BuildRetrievalPairs(opts.image_path,
                            opts.retrieval.num,
                            opts.retrieval.batch_size,
                            opts.retrieval.period,
                            opts.retrieval.similarity_threshold,
                            opts.retrieval.relative_threshold,
                            image_ids);
    metrics.retrieval_ms.AddSample(
        std::chrono::duration<double, std::milli>(
            Clock::now() - retrieval_start)
            .count());
    MergeUniquePairs(retrieval_pairs, &pairs);
  }
  metrics.pair_prepare_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                pair_prepare_start)
          .count());
  metrics.scheduled_pairs = static_cast<int>(pairs.size());

    // Resolve LightGlue batch size before engine init so we can load the
    // smallest engine bucket that supports the chosen batch.
    const int desc_dim = 128;
    const size_t kpt_slot_bytes =
      static_cast<size_t>(kPhoenixMaxKeypoints) * 2 * sizeof(float);
    const size_t desc_slot_bytes =
      static_cast<size_t>(kPhoenixMaxKeypoints) * desc_dim * sizeof(float);
    const size_t per_pair_staging = 2 * (kpt_slot_bytes + desc_slot_bytes);
  const int selected_batch = kPhoenixFixedBatchSize;

  const auto engine_init_start = Clock::now();
  const ResourceSnapshot before_engine = CaptureResourceSnapshot();
  LogResourceSnapshot("match.engine.before_init", before_engine);
    const auto lightglue_engine =
      sfm_phoenix::EnsurePhoenixLightGlueEngine(selected_batch);

  sfm_phoenix::LightGlueConfig matcher_config;
  matcher_config.engine_path = lightglue_engine.string();
  matcher_config.max_matches = opts.max_matches;

  sfm_phoenix::LightGlueMatcher matcher;
  if (!matcher.init(matcher_config)) {
    spdlog::error("Failed to initialize Phoenix LightGlue matcher");
    return EXIT_FAILURE;
  }
  metrics.engine_init_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                engine_init_start)
          .count());
  const ResourceSnapshot after_engine = CaptureResourceSnapshot();
  LogResourceSnapshot("match.engine.after_init",
                      after_engine,
                      &before_engine,
                      "engine=" + lightglue_engine.filename().string());

  std::unordered_map<colmap::image_t, CachedFeatures> feature_cache;
  feature_cache.reserve(ordered_images.size());
  ResourceSnapshot previous_snapshot = after_engine;

  // ─── VRAM-based batch sizing for GPU feature staging ──────────────────────
  // Each staging slot holds one pair's features (kpts + descs, both images)
  // pre-uploaded to GPU. Batch inference runs back-to-back without H2D stalls,
  // and geometry estimation tasks from a batch run in parallel on CPU threads
  // while the GPU processes the next batch.
  const int engine_max_batch = std::max(1, matcher.max_batch_size());
    const int max_kpts_per_img = matcher.max_keypoints();
  const int batch_size = std::min(selected_batch, engine_max_batch);
  size_t free_vram_bytes = 0, total_vram_bytes = 0;
  cudaMemGetInfo(&free_vram_bytes, &total_vram_bytes);
  spdlog::info(
      "Phoenix matching: free_vram={:.1f}MB per_pair_staging={:.1f}MB "
      "engine_max_batch={} fixed_batch_size={}",
      static_cast<double>(free_vram_bytes) / (1 << 20),
      static_cast<double>(per_pair_staging) / (1 << 20),
      engine_max_batch,
      batch_size);

  // GPU staging slots: pre-allocated device buffers for batch_size pairs.
  struct PairGpuSlot {
    sfm_phoenix::CudaBuffer kpts0;
    sfm_phoenix::CudaBuffer desc0;
    sfm_phoenix::CudaBuffer kpts1;
    sfm_phoenix::CudaBuffer desc1;
    int N0 = 0;
    int N1 = 0;
    bool valid = false;   // true iff H2D data was uploaded for this slot
    colmap::image_t id1 = 0;
    colmap::image_t id2 = 0;

    void Allocate(size_t kpt_bytes, size_t desc_bytes) {
      kpts0.resize(kpt_bytes);
      desc0.resize(desc_bytes);
      kpts1.resize(kpt_bytes);
      desc1.resize(desc_bytes);
    }
  };
  std::vector<PairGpuSlot> gpu_slots(batch_size);
  for (auto& s : gpu_slots) {
    s.Allocate(kpt_slot_bytes, desc_slot_bytes);
  }
  cudaStream_t upload_stream = nullptr;
  cudaStreamCreate(&upload_stream);

  // ─────────────────────────────────────────────────────────────────────────

  // Pipeline state: geometry estimation tasks run on CPU threads in parallel
  // with GPU inferences. A deque of depth batch_size allows geometry for a
  // full batch to overlap with the next batch's GPU work.
  struct PendingGeometry {
    colmap::image_t id1;
    colmap::image_t id2;
    colmap::FeatureMatches colmap_matches;
    std::future<colmap::TwoViewGeometry> geometry_future;
  };
  std::deque<PendingGeometry> pending_queue;
  bool interrupted = false;

  const auto flush_pending = [&](PendingGeometry& pg) {
    auto geometry = pg.geometry_future.get();
    const auto db_write_start = Clock::now();
    colmap::DatabaseTransaction transaction(database.get());
    database->WriteMatches(pg.id1, pg.id2, pg.colmap_matches);
    database->WriteTwoViewGeometry(pg.id1, pg.id2, geometry);
    metrics.db_write_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  db_write_start)
            .count());
  };

  // Process pairs in batches of batch_size.
  // Phase A (per batch): load features from DB/cache + async H2D upload.
  // Phase B (per batch): single TRT forward pass for all pairs in the batch
  //   (match_gpu_batch), then geometry tasks launched async so they overlap
  //   with the next batch's Phase A + B.
  int pair_idx = 0;
  const int total_pairs = static_cast<int>(pairs.size());

  while (pair_idx < total_pairs && !interrupted) {
    // ── Phase A: collect batch_size work items, upload features to GPU ──
    // BatchItem describes one pair's processing in this batch.
    struct BatchItem {
      colmap::image_t id1;
      colmap::image_t id2;
      int slot_idx;   // index into gpu_slots, or -1 if no GPU work needed
      bool skipped;   // pair was skipped (already in DB)
    };
    std::vector<BatchItem> batch;
    batch.reserve(batch_size);
    int slot_count = 0;

    while (slot_count < batch_size && pair_idx < total_pairs) {
      if (IsInterruptRequested()) {
        interrupted = true;
        break;
      }
      const auto [image_id1, image_id2] = pairs[pair_idx++];
      const bool has_matches =
          database->ExistsMatches(image_id1, image_id2);
      const bool has_geometry =
          database->ExistsInlierMatches(image_id1, image_id2);

      if ((has_matches || has_geometry) && opts.skip_existing &&
          !opts.overwrite_existing) {
        spdlog::info("Skip existing pair: {} <-> {}",
                     images_by_id.at(image_id1).Name(),
                     images_by_id.at(image_id2).Name());
        ++metrics.num_skipped_pairs;
        batch.push_back({image_id1, image_id2, -1, true});
        continue;
      }

      if (opts.overwrite_existing) {
        const auto db_write_start = Clock::now();
        colmap::DatabaseTransaction transaction(database.get());
        if (has_geometry) {
          database->DeleteInlierMatches(image_id1, image_id2);
        }
        if (has_matches) {
          database->DeleteMatches(image_id1, image_id2);
        }
        metrics.db_write_ms.AddSample(
            std::chrono::duration<double, std::milli>(Clock::now() -
                                                      db_write_start)
                .count());
      } else if (has_matches || has_geometry) {
        ++metrics.num_skipped_pairs;
        batch.push_back({image_id1, image_id2, -1, true});
        continue;
      }

      const auto& features1 =
          LoadFeatures(*database, images_by_id, cameras_by_id,
                       extraction_max_edge, image_id1,
                       &feature_cache, &metrics);
      const auto& features2 =
          LoadFeatures(*database, images_by_id, cameras_by_id,
                       extraction_max_edge, image_id2,
                       &feature_cache, &metrics);

      if (features1.num_keypoints == 0 || features2.num_keypoints == 0) {
        ++metrics.num_matched_pairs;
        ++metrics.num_zero_match_pairs;
        batch.push_back({image_id1, image_id2, -1, false});
        continue;
      }

      // Async H2D upload of this pair's features to the staging slot.
      auto& slot = gpu_slots[slot_count];
      slot.id1 = image_id1;
      slot.id2 = image_id2;
      slot.N0 = features1.num_keypoints;
      slot.N1 = features2.num_keypoints;
      slot.valid = true;
      cudaMemcpyAsync(slot.kpts0.ptr, features1.keypoints.data(),
                      static_cast<size_t>(slot.N0) * 2 * sizeof(float),
                      cudaMemcpyHostToDevice, upload_stream);
      cudaMemcpyAsync(slot.desc0.ptr, features1.descriptors.data(),
                      static_cast<size_t>(slot.N0) * desc_dim * sizeof(float),
                      cudaMemcpyHostToDevice, upload_stream);
      cudaMemcpyAsync(slot.kpts1.ptr, features2.keypoints.data(),
                      static_cast<size_t>(slot.N1) * 2 * sizeof(float),
                      cudaMemcpyHostToDevice, upload_stream);
      cudaMemcpyAsync(slot.desc1.ptr, features2.descriptors.data(),
                      static_cast<size_t>(slot.N1) * desc_dim * sizeof(float),
                      cudaMemcpyHostToDevice, upload_stream);
      batch.push_back({image_id1, image_id2, slot_count, false});
      ++slot_count;
    }

    // Wait for all H2D transfers in this batch to complete.
    if (slot_count > 0) {
      cudaStreamSynchronize(upload_stream);
    }

    // ── Phase B: single TRT forward pass for all pairs in this batch ──
    // Collect GPU inputs for batch inference.
    std::vector<sfm_phoenix::BatchMatchInput> trt_inputs;
    trt_inputs.reserve(slot_count);

    for (int j = 0; j < static_cast<int>(batch.size()); ++j) {
      const auto& item = batch[j];
      if (item.skipped || item.slot_idx < 0) continue;
      auto& slot = gpu_slots[item.slot_idx];
      trt_inputs.push_back({slot.kpts0.ptr, slot.desc0.ptr, slot.N0,
                             slot.kpts1.ptr, slot.desc1.ptr, slot.N1});
    }

    // One TRT forward pass for all pairs (or fast-path if count == 1).
    const auto match_start = Clock::now();
    auto trt_results = matcher.match_gpu_batch(trt_inputs);
    metrics.match_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - match_start)
            .count());

    // Process batch results in order: launch geometry async, log, etc.
    int ri = 0;
    for (const auto& item : batch) {
      if (item.skipped || item.slot_idx < 0) continue;

      const auto& matches = trt_results[ri++];

      // Flush the oldest pending geometry entry if the queue is at capacity.
      if (static_cast<int>(pending_queue.size()) >= batch_size) {
        flush_pending(pending_queue.front());
        pending_queue.pop_front();
      }

      if (matches.num_matches > 0) {
        const auto colmap_matches = sfm_phoenix::ToColmapMatches(matches);

        const auto& feat1 = feature_cache.at(item.id1);
        const auto& feat2 = feature_cache.at(item.id2);
        const auto& cam1 =
            cameras_by_id.at(images_by_id.at(item.id1).CameraId());
        const auto& cam2 =
            cameras_by_id.at(images_by_id.at(item.id2).CameraId());
        const auto& pts1 = feat1.points;
        const auto& pts2 = feat2.points;

        auto geometry_future = std::async(
            std::launch::async,
            [&cam1, &cam2, &pts1, &pts2, colmap_matches, &metrics]() {
              const auto geometry_start = Clock::now();
              colmap::TwoViewGeometryOptions geometry_options;
              auto geometry = colmap::EstimateTwoViewGeometry(
                  cam1, pts1, cam2, pts2, colmap_matches, geometry_options);
              metrics.geometry_ms.AddSample(
                  std::chrono::duration<double, std::milli>(Clock::now() -
                                                            geometry_start)
                      .count());
              return geometry;
            });

        pending_queue.push_back(PendingGeometry{
            item.id1, item.id2,
            std::move(colmap_matches),
            std::move(geometry_future)});
      } else {
        ++metrics.num_zero_match_pairs;
      }
      ++metrics.num_matched_pairs;
      metrics.cached_images = static_cast<int>(feature_cache.size());

      spdlog::info("Matched {} <-> {}: {}",
                   images_by_id.at(item.id1).Name(),
                   images_by_id.at(item.id2).Name(),
                   matches.num_matches);
      const ResourceSnapshot current_snapshot = CaptureResourceSnapshot();
      LogResourceSnapshot(
          "match.pair.done",
          current_snapshot,
          &previous_snapshot,
          "pair=" + images_by_id.at(item.id1).Name() + "<->" +
              images_by_id.at(item.id2).Name() +
              " matches=" + std::to_string(matches.num_matches));
      previous_snapshot = current_snapshot;

      gpu_slots[item.slot_idx].valid = false;
    }

    if (IsInterruptRequested()) {
      interrupted = true;
    }
  }

  // Flush all remaining pending geometry results.
  while (!pending_queue.empty()) {
    flush_pending(pending_queue.front());
    pending_queue.pop_front();
  }

  cudaStreamDestroy(upload_stream);

  metrics.wall_total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - matching_start)
          .count();

  spdlog::info("Phoenix matching finished. matched_pairs={} skipped_pairs={}",
               metrics.num_matched_pairs,
               metrics.num_skipped_pairs);
  LogFeatureMatchingMetrics(metrics);
  if (interrupted) {
    spdlog::warn("Phoenix matching interrupted by Ctrl+C");
    return EXIT_FAILURE;
  }
  return EXIT_SUCCESS;
}

}  // namespace

int ExecFeatureMatcher(const std::string& database_path,
                       const FeatureMatchingOptions& opts) {
  constexpr int kMaxMatchedFeatures = 4000;
  if (opts.linear_overlap_num < 0 || opts.quadratic_overlap_num < 0) {
    spdlog::error("Phoenix overlap params must be non-negative");
    return EXIT_FAILURE;
  }
  if (opts.retrieval.num < 0 || opts.retrieval.batch_size <= 0) {
    spdlog::error("Phoenix retrieval params must be positive/zero");
    return EXIT_FAILURE;
  }
  if (opts.retrieval.period <= 0) {
    spdlog::error(
        "SequentialMatching.loop_detection_period must be positive");
    return EXIT_FAILURE;
  }
  if (opts.retrieval.similarity_threshold < 0.0 ||
      opts.retrieval.similarity_threshold > 1.0) {
    spdlog::error(
        "Phoenix.retrieval_similarity_threshold must be in [0, 1]");
    return EXIT_FAILURE;
  }
  if (opts.retrieval.relative_threshold < 0.0 ||
      opts.retrieval.relative_threshold > 1.0) {
    spdlog::error(
        "Phoenix.retrieval_relative_threshold must be in [0, 1]");
    return EXIT_FAILURE;
  }
  FeatureMatchingOptions clamped = opts;
  if (clamped.max_matches > kMaxMatchedFeatures) {
    spdlog::warn("Phoenix.max_matches={} exceeds hard cap {}, clamping",
                 clamped.max_matches,
                 kMaxMatchedFeatures);
    clamped.max_matches = kMaxMatchedFeatures;
  }
  return DoMatching(database_path, clamped);
}

int RunFeatureMatcher(int argc, char** argv) {
  FeatureMatchingOptions cli_options;

  colmap::OptionManager options;
  AddFeatureMatcherOptions(&options, &cli_options);
  options.Parse(argc, argv);

  return ExecFeatureMatcher(*options.database_path, cli_options);
}

}  // namespace phoenix_tool