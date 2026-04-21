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
#include <filesystem>
#include <fstream>
#include <future>
#include <optional>
#include <unordered_set>

namespace phoenix_tool {

namespace {

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

    for (int ki = 0; ki < k; ++ki) {
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
    const std::filesystem::path& image_list_path,
    const std::filesystem::path& retrieval_model_path,
    const int retrieval_num,
    const int retrieval_batch_size,
    const std::unordered_map<std::string, colmap::image_t>& image_ids) {
  if (retrieval_num <= 0) return {};
  if (image_path.empty()) {
    throw std::runtime_error(
        "Phoenix retrieval requires --image_path when "
        "Phoenix.retrieval_num > 0");
  }

  const auto onnx_path = retrieval_model_path.empty()
                             ? DefaultRetrievalModelPath()
                             : retrieval_model_path;

  const auto t_engine_start = std::chrono::steady_clock::now();
  const auto engine_path = sfm_phoenix::EnsureRetrievalEngine(onnx_path);

  spdlog::info(
      "Phoenix retrieval: generating top-{} pairs via TRT engine {}",
      retrieval_num, engine_path.string());

  sfm_phoenix::TrtEngine engine;
  if (!engine.load(engine_path.string())) {
    throw std::runtime_error("Failed to load retrieval engine: " +
                             engine_path.string());
  }
  const double engine_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_engine_start)
          .count();

  // Collect image names — ordered or from DB
  std::vector<std::string> image_names;
  if (!image_list_path.empty()) {
    for (const auto& name : ReadTextLines(image_list_path)) {
      if (image_ids.count(name)) image_names.push_back(name);
    }
  } else {
    for (const auto& [name, id] : image_ids) image_names.push_back(name);
    std::sort(image_names.begin(), image_names.end());
  }
  if (image_names.empty()) return {};

  const auto t_embed_start = std::chrono::steady_clock::now();
  const auto embeddings = ExtractDinoEmbeddings(
      image_names, image_path, engine, retrieval_batch_size);
  const double embed_ms =
      std::chrono::duration<double, std::milli>(
          std::chrono::steady_clock::now() - t_embed_start)
          .count();

  const auto t_topk_start = std::chrono::steady_clock::now();
  auto pairs =
      TopKRetrievalPairs(image_names, embeddings, retrieval_num, image_ids);
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

}  // namespace

int RunFeatureMatcher(int argc, char** argv) {
  constexpr int kMaxMatchedFeatures = 4000;
  using Clock = std::chrono::steady_clock;

  std::filesystem::path pair_list_path;
  std::filesystem::path image_path;
  std::filesystem::path image_list_path;
  std::filesystem::path retrieval_model_path;
  int max_edge = 1600;
  int max_matches = kMaxMatchedFeatures;
  int retrieval_num = 50;
  int retrieval_batch_size = 32;
  int linear_overlap_num = 10;
  int quadratic_overlap_num = 10;
  bool skip_existing = true;
  bool overwrite_existing = false;

  colmap::OptionManager options;
  options.AddDatabaseOptions();
  options.AddDefaultOption("pair_list_path", &pair_list_path);
  options.AddDefaultOption("image_path", &image_path);
  options.AddDefaultOption("image_list_path", &image_list_path);
  options.AddDefaultOption("Phoenix.retrieval_model_path",
                           &retrieval_model_path);
  options.AddDefaultOption("Phoenix.max_edge", &max_edge);
  options.AddDefaultOption("Phoenix.max_matches", &max_matches);
  options.AddDefaultOption("Phoenix.retrieval_num", &retrieval_num);
  options.AddDefaultOption("Phoenix.retrieval_batch_size",
                           &retrieval_batch_size);
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
  if (retrieval_num < 0 || retrieval_batch_size <= 0) {
    spdlog::error("Phoenix retrieval params must be positive/zero");
    return EXIT_FAILURE;
  }

  if (max_matches > kMaxMatchedFeatures) {
    spdlog::warn("Phoenix.max_matches={} exceeds hard cap {}, clamping",
                 max_matches,
                 kMaxMatchedFeatures);
    max_matches = kMaxMatchedFeatures;
  }

  FeatureMatchingMetrics metrics;
  const auto matching_start = Clock::now();

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

  const auto pair_prepare_start = Clock::now();
  std::vector<std::pair<colmap::image_t, colmap::image_t>> pairs;
  if (pair_list_path.empty()) {
    if (linear_overlap_num > 0 || quadratic_overlap_num > 0) {
      pairs = BuildSequentialPairs(ordered_images,
                                   linear_overlap_num,
                                   quadratic_overlap_num);
    } else if (retrieval_num <= 0) {
      pairs = BuildExhaustivePairs(ordered_images);
    }
    if (retrieval_num > 0) {
      const auto resolved_model = retrieval_model_path.empty()
                                      ? DefaultRetrievalModelPath()
                                      : retrieval_model_path;
      const auto retrieval_start = Clock::now();
      const auto retrieval_pairs = BuildRetrievalPairs(image_path,
                                                       image_list_path,
                                                       resolved_model,
                                                       retrieval_num,
                                                       retrieval_batch_size,
                                                       image_ids);
      metrics.retrieval_ms.AddSample(
          std::chrono::duration<double, std::milli>(
              Clock::now() - retrieval_start)
              .count());
      MergeUniquePairs(retrieval_pairs, &pairs);
    }
  } else {
    pairs = ReadPairs(pair_list_path, image_ids);
  }
  metrics.pair_prepare_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                pair_prepare_start)
          .count());
  metrics.scheduled_pairs = static_cast<int>(pairs.size());

  const auto engine_init_start = Clock::now();
  const auto lightglue_engine = sfm_phoenix::EnsurePhoenixLightGlueEngine();

  sfm_phoenix::LightGlueConfig matcher_config;
  matcher_config.engine_path = lightglue_engine.string();
  matcher_config.max_matches = max_matches;

  sfm_phoenix::LightGlueMatcher matcher;
  if (!matcher.init(matcher_config)) {
    spdlog::error("Failed to initialize Phoenix LightGlue matcher");
    return EXIT_FAILURE;
  }
  metrics.engine_init_ms.AddSample(
      std::chrono::duration<double, std::milli>(Clock::now() -
                                                engine_init_start)
          .count());

  std::unordered_map<colmap::image_t, CachedFeatures> feature_cache;
  feature_cache.reserve(ordered_images.size());

  // Pipeline state: geometry estimation for previous pair runs on a CPU
  // thread while the GPU processes the next match.
  struct PendingGeometry {
    colmap::image_t id1;
    colmap::image_t id2;
    colmap::FeatureMatches colmap_matches;
    std::future<colmap::TwoViewGeometry> geometry_future;
  };
  std::optional<PendingGeometry> pending;

  const auto flush_pending = [&](PendingGeometry& pg) {
    auto geometry = pg.geometry_future.get();
    const auto db_write_start = Clock::now();
    database->WriteMatches(pg.id1, pg.id2, pg.colmap_matches);
    database->WriteTwoViewGeometry(pg.id1, pg.id2, geometry);
    metrics.db_write_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() -
                                                  db_write_start)
            .count());
  };

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
      ++metrics.num_skipped_pairs;
      continue;
    }

    if (overwrite_existing) {
      const auto db_write_start = Clock::now();
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
      continue;
    }

    const auto& features1 = LoadFeatures(*database,
                                         images_by_id,
                                         cameras_by_id,
                                         max_edge,
                                         image_id1,
                                         &feature_cache,
                                         &metrics);
    const auto& features2 = LoadFeatures(*database,
                                         images_by_id,
                                         cameras_by_id,
                                         max_edge,
                                         image_id2,
                                         &feature_cache,
                                         &metrics);
    if (features1.num_keypoints == 0 || features2.num_keypoints == 0) {
      ++metrics.num_matched_pairs;
      continue;
    }

    const auto match_start = Clock::now();
    const auto matches = matcher.match(features1.keypoints.data(),
                                       features1.descriptors.data(),
                                       features1.num_keypoints,
                                       features2.keypoints.data(),
                                       features2.descriptors.data(),
                                       features2.num_keypoints);
    metrics.match_ms.AddSample(
        std::chrono::duration<double, std::milli>(Clock::now() - match_start)
            .count());

    // Flush previous pending geometry result (was running on CPU thread
    // in parallel with the GPU match above).
    if (pending) {
      flush_pending(*pending);
      pending.reset();
    }

    if (matches.num_matches > 0) {
      const auto colmap_matches = sfm_phoenix::ToColmapMatches(matches);

      // Launch geometry estimation on a background thread so the next
      // GPU match can start immediately.
      const auto& cam1 =
          cameras_by_id.at(images_by_id.at(image_id1).CameraId());
      const auto& cam2 =
          cameras_by_id.at(images_by_id.at(image_id2).CameraId());
      const auto& pts1 = features1.points;
      const auto& pts2 = features2.points;

      auto geometry_future =
          std::async(std::launch::async,
                     [&cam1, &cam2, &pts1, &pts2,
                      colmap_matches, &metrics]() {
                       const auto geometry_start = Clock::now();
                       colmap::TwoViewGeometryOptions geometry_options;
                       auto geometry = colmap::EstimateTwoViewGeometry(
                           cam1, pts1, cam2, pts2,
                           colmap_matches, geometry_options);
                       metrics.geometry_ms.AddSample(
                           std::chrono::duration<double, std::milli>(
                               Clock::now() - geometry_start)
                               .count());
                       return geometry;
                     });

      pending = PendingGeometry{
          image_id1, image_id2,
          std::move(colmap_matches),
          std::move(geometry_future)};
    } else {
      ++metrics.num_zero_match_pairs;
    }
    ++metrics.num_matched_pairs;
    metrics.cached_images = static_cast<int>(feature_cache.size());

    spdlog::info("Matched {} <-> {}: {}",
                 images_by_id.at(image_id1).Name(),
                 images_by_id.at(image_id2).Name(),
                 matches.num_matches);
  }
  // Flush last pending geometry result.
  if (pending) {
    flush_pending(*pending);
    pending.reset();
  }
  database->EndTransaction();

  metrics.wall_total_ms =
      std::chrono::duration<double, std::milli>(Clock::now() - matching_start)
          .count();

  spdlog::info("Phoenix matching finished. matched_pairs={} skipped_pairs={}",
               metrics.num_matched_pairs,
               metrics.num_skipped_pairs);
  LogFeatureMatchingMetrics(metrics);
  return EXIT_SUCCESS;
}

}  // namespace phoenix_tool