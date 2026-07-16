
#include <colmap/scene/database.h>
#include <colmap/scene/reconstruction.h>

#include <algorithm>
#include <filesystem>
#include <map>
#include <optional>
#include <random>
#include <unordered_map>

#include <omp.h>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

// #include "common/types.h"
// #include "core/utils.h"
// #include "core/xcolor_lib.h"
// #include "io/colmap_io.h"
#include "migration/string.h"
#include "migration/utils.h"
#include "xcolor_lib.h"

DEFINE_string(images_path,
              R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\cubemap_colmap\images)",
              "Images path");
DEFINE_string(sfm_result_path,
              R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\cubemap_colmap\sparse)",
              "SFM databaset filename");
DEFINE_string(point_cloud_filename,
              R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\small_plane_refined.las)",
              "Point cloud filename (LAS format)");
DEFINE_string(output_path,
              R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output)",
              "Output path");
DEFINE_string(mask_path,
              "",
              "Optional masks path. If empty, uses sibling masks directory "
              "when it exists.");
DEFINE_int32(max_color_candidates,
             xcolor::kColorInlierMaxNum,
             "Maximum nearest color candidates retained per point.");

namespace {

void InitPlaintextSpdLog() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>("", console_sink);
  spdlog::set_default_logger(logger);
}

}  // namespace

namespace xcolor {

namespace fs = std::filesystem;

namespace {

std::string NormalizeRelativePath(const std::string& path) {
  std::string normalized = path;
  std::replace(normalized.begin(), normalized.end(), '\\', '/');
  return normalized;
}

std::optional<fs::path> FindMaskFile(const fs::path& masks_root,
                                     const std::string& image_name) {
  if (masks_root.empty() || !fs::exists(masks_root)) {
    return std::nullopt;
  }

  const fs::path relative_image_path = fs::path(image_name);
  const std::vector<std::string> extensions = {
      relative_image_path.extension().string(), ".png", ".jpg", ".jpeg"};

  for (const std::string& extension : extensions) {
    fs::path relative_mask_path = relative_image_path;
    if (!extension.empty()) {
      relative_mask_path.replace_extension(extension);
    }
    const fs::path candidate = masks_root / relative_mask_path;
    if (fs::exists(candidate)) {
      return candidate;
    }
  }

  return std::nullopt;
}

}  // namespace

pcl::PointCloud<pcl::PointXYZRGB> ReadPointCloudFromLAS(
    const std::string& filename) {
  spdlog::info("Reading point cloud from LAS file: {}", filename);

  pcl::PointCloud<pcl::PointXYZRGB> cloud;

  // Use PDAL to read LAS file
  pdal::StageFactory factory;
  pdal::Stage* reader = factory.createStage("readers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view = *viewSet.begin();

  spdlog::info("Converting {} points from LAS to PCL format", view->size());

  // Convert PDAL points to PCL format
  cloud.reserve(view->size());
  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZRGB point;
    point.x = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    point.y = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    point.z = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    cloud.push_back(point);
  }

  spdlog::info("Loaded {} points from LAS file", cloud.size());
  return cloud;
}

void ReadImages(const std::string& sfm_path,
                const std::string& images_path,
                const std::string& mask_path,
                std::vector<Image>& images) {
  const fs::path images_root(images_path);
  const fs::path cubemap_root = images_root.parent_path();
  const fs::path depths_root = cubemap_root / "depths";
  const fs::path masks_root =
      mask_path.empty() ? cubemap_root / "masks" : fs::path(mask_path);
  const bool use_masks = fs::exists(masks_root);
  if (use_masks) {
    spdlog::info("Loading masks from {} ...", masks_root.string());
  }

  colmap::Reconstruction rec;
  rec.ReadBinary(sfm_path);

  spdlog::info("Loading image poses from {} ...", sfm_path + "/images.bin");
  auto& raw_images = rec.Images();
  spdlog::info("Load {} image poses.", raw_images.size());

  // todo kk to be removed
  // std::unordered_map<colmap::image_t, colmap::Image> raw_images_filtered;
  // {
  //   std::map<colmap::camera_t, std::vector<colmap::Image>> cam_to_images;
  //   for (const auto& [image_id, image] : raw_images) {
  //     cam_to_images[image.CameraId()].push_back(image);
  //   }
  //   for (auto& [cam_id, images_vec] : cam_to_images) {
  //     spdlog::info("camera id {} has {} images.", cam_id, images_vec.size());
  //     std::sort(images_vec.begin(),
  //               images_vec.end(),
  //               [](const colmap::Image& a, const colmap::Image& b) {
  //                 return a.Name() < b.Name();
  //               });
  //     // images_vec.erase(images_vec.begin() + 100, images_vec.end());
  //     std::vector<colmap::Image> images_vec_filtered;
  //     int count = 0;
  //     for (auto& e : images_vec) {
  //       if (++count % 4 == 0) {
  //         images_vec_filtered.push_back(e);
  //       }
  //     }
  //     images_vec.swap(images_vec_filtered);
  //   }
  //   for (auto& [cam_id, images_vec] : cam_to_images) {
  //     for (auto& image : images_vec) {
  //       raw_images_filtered[image.ImageId()] = image;
  //     }
  //   }
  // }

  spdlog::info("Loading cameras from {} ...", sfm_path + "/cameras.bin");
  auto& raw_cameras = rec.Cameras();
  spdlog::info("Load {} cameras.", raw_cameras.size());

  spdlog::info("Loading images from {} ...", images_path);
  images.clear();
  for (auto& [image_id, raw_image] : raw_images) {
  // for (auto& [image_id, raw_image] : raw_images_filtered) {
    Image image;
    image.filename = images_path + "/" + raw_image.Name();
    image.pose = raw_image.CamFromWorld();
    image.camera = raw_cameras.at(raw_image.CameraId());

    const std::string relative_depth_name = NormalizeRelativePath(
        fs::path(raw_image.Name()).replace_extension(".png").generic_string());
    const fs::path depth_filename = depths_root / fs::path(relative_depth_name);
    if (!fs::exists(depth_filename)) {
      spdlog::warn("Missing depth image for {}", raw_image.Name());
    } else {
      image.depth_filename = depth_filename.string();
      image.depth_intrinsics = DepthIntrinsicsFromCamera(image.camera);
      image.has_depth = true;
    }
    if (use_masks) {
      const auto mask_file = FindMaskFile(masks_root, raw_image.Name());
      if (mask_file.has_value()) {
        image.mask_filename = mask_file->string();
        image.has_mask = true;
      } else {
        spdlog::warn("Missing mask for image {}", raw_image.Name());
      }
    }
    images.push_back(image);
  }
  spdlog::info("Load {} images.", images.size());
}

}  // namespace xcolor

int main(int argc, char** argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  if (FLAGS_max_color_candidates < 1) {
    spdlog::error("--max_color_candidates must be >= 1");
    return 1;
  }
  spdlog::set_level(spdlog::level::debug);

  InitPlaintextSpdLog();

  int cores = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  spdlog::info("Using {}/{} cores.", cores_used, cores);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  std::vector<xcolor::Image> images;

  // loading images and camera parameters
  PrintMemoryUsage();
  xcolor::ReadImages(
      FLAGS_sfm_result_path, FLAGS_images_path, FLAGS_mask_path, images);
  PrintMemoryUsage();

  // loading point cloud
  spdlog::info("Loading point cloud from {} ...", FLAGS_point_cloud_filename);
  pcl::PointCloud<pcl::PointXYZRGB> cloud =
      xcolor::ReadPointCloudFromLAS(FLAGS_point_cloud_filename);
  PrintMemoryUsage();
  if (cloud.empty()) {
    spdlog::error("Load point cloud failed: {}", FLAGS_point_cloud_filename);
    exit(1);
  }
  spdlog::info("Load {} points.", cloud.size());

  spdlog::info("Start xcolor...");
  xcolor::PerformXColor(
      images, cloud, FLAGS_output_path, FLAGS_max_color_candidates);
  spdlog::info("Finish xcolor.");

  return 0;
}
