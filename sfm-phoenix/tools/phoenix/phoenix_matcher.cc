#include "phoenix_tool.h"

#include "sfm_phoenix/bridges/colmap_bridge.h"
#include "sfm_phoenix/internal/model_runtime.h"
#include "sfm_phoenix/matchers/lightglue.h"

#include <colmap/controllers/option_manager.h>
#include <colmap/estimators/two_view_geometry.h>

#include <spdlog/spdlog.h>

#include <chrono>
#include <future>

namespace phoenix_tool {

namespace {

void LogFeatureMatchingMetrics(const FeatureMatchingMetrics& metrics) {
  const double pairs_per_sec =
      metrics.wall_total_ms > 0.0
          ? (1000.0 * metrics.num_matched_pairs) / metrics.wall_total_ms
          : 0.0;

  spdlog::info(
      "Phoenix matching performance: scheduled_pairs={} matched_pairs={} "
      "skipped_pairs={} zero_match_pairs={} cached_images={} "
      "feature_cache_hits={} feature_cache_misses={} pair_prepare_ms={:.3f} "
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
    } else {
      pairs = BuildExhaustivePairs(ordered_images);
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