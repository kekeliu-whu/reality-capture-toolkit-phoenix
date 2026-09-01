
#include <colmap/scene/database.h>
#include <colmap/scene/reconstruction.h>

#include <algorithm>
#include <array>
#include <cctype>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <random>
#include <unordered_map>

#include <omp.h>
#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>
#include <opencv2/opencv.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>

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
              "Point cloud filename (LAS, LAZ, PCD, or PLY format)");
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
DEFINE_bool(generate_fisheye_depths,
            false,
            "Render depth directly in each original OPENCV_FISHEYE view on "
            "the GPU instead of loading cubemap depth images.");
DEFINE_bool(gpu_visibility,
            true,
            "Reuse CUDA projection/depth results for color visibility instead "
            "of testing every point again on the CPU.");
DEFINE_bool(gpu_color_fusion,
            true,
            "Sample depth-visible fisheye colors, reject multi-view color "
            "outliers, and select from three sharpness-aware real-view "
            "hypotheses directly on the GPU without downloading a "
            "visible-point list for every image.");
DEFINE_bool(gpu_color_smooth_fusion,
            true,
            "Use a spatially coherent robust mean of agreeing visible views "
            "instead of selecting one source pixel per point.");
DEFINE_double(fisheye_depth_scale,
              0.25,
              "Generated fisheye depth resolution relative to the source image.");
DEFINE_double(depth_voxel_size,
              0.03,
              "Point splat voxel size in meters for generated fisheye depth.");
DEFINE_double(depth_visibility_tolerance,
              xcolor::kDepthVisibilityTolerance,
              "Maximum generated-depth discrepancy in meters for accepting "
              "a color observation.");
DEFINE_double(depth_max_distance,
              30.0,
              "Maximum generated depth distance in meters; <=0 disables it.");
DEFINE_int32(gpu_chunk_points,
             3000000,
             "Maximum point count in each CUDA fisheye depth chunk.");
DEFINE_string(generated_depth_path,
              "",
              "Optional directory for saving generated fisheye depth PNGs. "
              "Depths are used directly from memory when empty.");
DEFINE_int32(image_step,
             1,
             "Use every Nth image independently within each camera directory.");
DEFINE_int32(limit_images,
             0,
             "Optional total image limit for smoke tests; 0 processes all.");

namespace {

void InitPlaintextSpdLog() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger       = std::make_shared<spdlog::logger>("", console_sink);
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

  const fs::path relative_image_path        = fs::path(image_name);
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

  // Direct fisheye coloring commonly uses one fixed mask per physical camera,
  // e.g. masks_root/cam0.png for every cam0/*.jpg image.
  const auto first_component = relative_image_path.begin();
  if (first_component != relative_image_path.end()) {
    fs::path fixed_mask = masks_root / *first_component;
    fixed_mask.replace_extension(".png");
    if (fs::exists(fixed_mask)) {
      return fixed_mask;
    }
  }

  return std::nullopt;
}

bool HasNamedFloatNormalExtraBytes(const std::string& filename) {
  std::ifstream input(filename, std::ios::binary);
  std::array<unsigned char, 227> header{};
  if (!input.read(reinterpret_cast<char*>(header.data()), header.size()) ||
      std::string(reinterpret_cast<const char*>(header.data()), 4) != "LASF") {
    return false;
  }
  const auto read_u16 = [&](size_t offset) {
    return static_cast<uint16_t>(header[offset]) |
           (static_cast<uint16_t>(header[offset + 1]) << 8);
  };
  const auto read_u32 = [&](size_t offset) {
    return static_cast<uint32_t>(header[offset]) |
           (static_cast<uint32_t>(header[offset + 1]) << 8) |
           (static_cast<uint32_t>(header[offset + 2]) << 16) |
           (static_cast<uint32_t>(header[offset + 3]) << 24);
  };
  constexpr std::array<uint16_t, 11> kBaseRecordLengths = {
      20, 28, 26, 34, 57, 63, 30, 36, 38, 59, 67};
  const uint8_t point_format   = header[104] & 0x3f;
  const uint16_t record_length = read_u16(105);
  if (point_format >= kBaseRecordLengths.size() ||
      record_length < kBaseRecordLengths[point_format] + 12) {
    return false;
  }
  const uint16_t header_size  = read_u16(94);
  const uint32_t point_offset = read_u32(96);
  if (point_offset <= header_size || point_offset - header_size > 16 * 1024 * 1024) {
    return false;
  }
  std::string vlr_bytes(point_offset - header_size, '\0');
  input.seekg(header_size);
  if (!input.read(vlr_bytes.data(), vlr_bytes.size())) {
    return false;
  }
  return vlr_bytes.find("NormalX") != std::string::npos &&
         vlr_bytes.find("NormalY") != std::string::npos &&
         vlr_bytes.find("NormalZ") != std::string::npos;
}

}  // namespace

pcl::PointCloud<pcl::PointXYZRGBNormal> ReadPointCloudFromLAS(
    const std::string& filename) {
  spdlog::info("Reading point cloud from LAS file: {}", filename);

  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;

  // Use PDAL to read LAS file
  pdal::StageFactory factory;
  pdal::Stage* reader = factory.createStage("readers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", PlatformToUTF8(filename)));
  if (HasNamedFloatNormalExtraBytes(filename)) {
    opts.add(pdal::Option(
        "extra_dims", "NormalX=float,NormalY=float,NormalZ=float"));
    spdlog::info("Reading named NormalX/NormalY/NormalZ LAS extra bytes");
  }
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  spdlog::info("Converting {} points from LAS to PCL format", view->size());

  const pdal::Dimension::Id normal_x_dim = view->layout()->findDim("NormalX");
  const pdal::Dimension::Id normal_y_dim = view->layout()->findDim("NormalY");
  const pdal::Dimension::Id normal_z_dim = view->layout()->findDim("NormalZ");
  const bool has_normals                 = normal_x_dim != pdal::Dimension::Id::Unknown &&
                                           normal_y_dim != pdal::Dimension::Id::Unknown &&
                                           normal_z_dim != pdal::Dimension::Id::Unknown;

  // Convert PDAL points to PCL format
  cloud.reserve(view->size());
  size_t valid_normal_count = 0;
  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZRGBNormal point;
    point.x = view->getFieldAs<double>(pdal::Dimension::Id::X, i);
    point.y = view->getFieldAs<double>(pdal::Dimension::Id::Y, i);
    point.z = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    if (has_normals) {
      point.normal_x = view->getFieldAs<float>(normal_x_dim, i);
      point.normal_y = view->getFieldAs<float>(normal_y_dim, i);
      point.normal_z = view->getFieldAs<float>(normal_z_dim, i);
      if (point.getNormalVector3fMap().allFinite() &&
          point.getNormalVector3fMap().squaredNorm() >= 1e-12f) {
        ++valid_normal_count;
      }
    } else {
      point.normal_x = std::numeric_limits<float>::quiet_NaN();
      point.normal_y = std::numeric_limits<float>::quiet_NaN();
      point.normal_z = std::numeric_limits<float>::quiet_NaN();
    }
    cloud.push_back(point);
  }

  spdlog::info("Loaded {} points from LAS file; {} have valid normals",
               cloud.size(),
               valid_normal_count);
  return cloud;
}

pcl::PointCloud<pcl::PointXYZRGBNormal> ReadPointCloud(
    const std::string& filename) {
  const fs::path path(filename);
  std::string extension = path.extension().string();
  std::transform(
      extension.begin(), extension.end(), extension.begin(), ::tolower);
  if (extension == ".las" || extension == ".laz") {
    return ReadPointCloudFromLAS(filename);
  }
  if (extension == ".pcd" || extension == ".ply") {
    pcl::PointCloud<pcl::PointNormal> source_cloud;
    int read_result = -1;
    if (extension == ".pcd") {
      spdlog::info("Reading point cloud from PCD file: {}", filename);
      read_result = pcl::io::loadPCDFile(filename, source_cloud);
    } else {
      spdlog::info("Reading point cloud from PLY file: {}", filename);
      read_result = pcl::io::loadPLYFile(filename, source_cloud);
    }
    if (read_result != 0) {
      throw std::runtime_error("Failed to load point-cloud file: " + filename);
    }
    pcl::PointCloud<pcl::PointXYZRGBNormal> cloud;
    cloud.reserve(source_cloud.size());
    size_t valid_normal_count = 0;
    for (const auto& source : source_cloud) {
      pcl::PointXYZRGBNormal point;
      point.x        = source.x;
      point.y        = source.y;
      point.z        = source.z;
      point.normal_x = source.normal_x;
      point.normal_y = source.normal_y;
      point.normal_z = source.normal_z;
      point.r = point.g = point.b = 0;
      if (point.getNormalVector3fMap().allFinite() &&
          point.getNormalVector3fMap().squaredNorm() >= 1e-12f) {
        ++valid_normal_count;
      }
      cloud.push_back(point);
    }
    spdlog::info("Loaded {} points from {} file; {} have valid normals",
                 cloud.size(),
                 extension == ".pcd" ? "PCD" : "PLY",
                 valid_normal_count);
    return cloud;
  }
  throw std::runtime_error("Unsupported point-cloud format: " + extension);
}

void ReadImages(const std::string& sfm_path,
                const std::string& images_path,
                const std::string& mask_path,
                std::vector<Image>& images) {
  const fs::path images_root(images_path);
  const fs::path cubemap_root = images_root.parent_path();
  const fs::path depths_root  = cubemap_root / "depths";
  const bool use_depth_files  = fs::exists(depths_root);
  const fs::path masks_root =
      mask_path.empty() ? cubemap_root / "masks" : fs::path(mask_path);
  const bool use_masks = fs::exists(masks_root);
  if (use_masks) {
    spdlog::info("Loading masks from {} ...", masks_root.string());
  }

  colmap::Reconstruction rec;
  // Accept both the text model emitted directly from ImgPose.txt and a binary
  // COLMAP model. Reconstruction::Read prefers binary files when both exist.
  rec.Read(sfm_path);

  const fs::path model_root(sfm_path);
  const fs::path image_model_path =
      fs::exists(model_root / "images.bin") ? model_root / "images.bin"
                                            : model_root / "images.txt";
  const fs::path camera_model_path =
      fs::exists(model_root / "cameras.bin") ? model_root / "cameras.bin"
                                             : model_root / "cameras.txt";
  spdlog::info("Loading image poses from {} ...", image_model_path.string());
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

  spdlog::info("Loading cameras from {} ...", camera_model_path.string());
  auto& raw_cameras = rec.Cameras();
  spdlog::info("Load {} cameras.", raw_cameras.size());

  spdlog::info("Loading images from {} ...", images_path);
  images.clear();
  for (auto& [image_id, raw_image] : raw_images) {
    // for (auto& [image_id, raw_image] : raw_images_filtered) {
    Image image;
    image.name     = NormalizeRelativePath(raw_image.Name());
    image.filename = images_path + "/" + raw_image.Name();
    image.pose     = raw_image.CamFromWorld();
    image.camera   = raw_cameras.at(raw_image.CameraId());

    const std::string relative_depth_name = NormalizeRelativePath(
        fs::path(raw_image.Name()).replace_extension(".png").generic_string());
    const fs::path depth_filename = depths_root / fs::path(relative_depth_name);
    if (!fs::exists(depth_filename)) {
      if (use_depth_files) {
        spdlog::warn("Missing depth image for {}", raw_image.Name());
      }
    } else {
      image.depth_filename = depth_filename.string();
      image.depth_camera   = image.camera;
      image.has_depth      = true;
    }
    if (use_masks) {
      const auto mask_file = FindMaskFile(masks_root, raw_image.Name());
      if (mask_file.has_value()) {
        image.mask_filename = mask_file->string();
        image.has_mask      = true;
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
  if (FLAGS_max_color_candidates < 1 ||
      FLAGS_max_color_candidates > xcolor::kColorInlierMaxNum) {
    spdlog::error("--max_color_candidates must be in [1, {}]",
                  xcolor::kColorInlierMaxNum);
    return 1;
  }
  if (!(FLAGS_fisheye_depth_scale > 0.0) ||
      !(FLAGS_depth_voxel_size > 0.0) ||
      FLAGS_depth_visibility_tolerance < 0.0 ||
      FLAGS_gpu_chunk_points <= 0 ||
      FLAGS_image_step <= 0 || FLAGS_limit_images < 0) {
    spdlog::error(
        "Invalid fisheye depth/image selection options: scale and voxel size "
        "must be positive, gpu_chunk_points/image_step must be positive, and "
        "limit_images must be non-negative");
    return 1;
  }
  spdlog::set_level(spdlog::level::debug);

  InitPlaintextSpdLog();

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  spdlog::info("Using {}/{} cores.", cores_used, cores);
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  std::vector<xcolor::Image> images;

  // loading images and camera parameters
  PrintMemoryUsage();
  xcolor::ReadImages(
      FLAGS_sfm_result_path, FLAGS_images_path, FLAGS_mask_path, images);
  std::sort(images.begin(), images.end(), [](const auto& lhs, const auto& rhs) {
    return lhs.name < rhs.name;
  });
  if (FLAGS_image_step > 1 || FLAGS_limit_images > 0) {
    std::vector<xcolor::Image> selected;
    std::unordered_map<std::string, int> camera_ordinals;
    for (auto& image : images) {
      const std::string camera_directory =
          std::filesystem::path(image.name).parent_path().generic_string();
      const int ordinal = camera_ordinals[camera_directory]++;
      if (ordinal % FLAGS_image_step != 0) {
        continue;
      }
      selected.push_back(std::move(image));
      if (FLAGS_limit_images > 0 &&
          selected.size() >= static_cast<size_t>(FLAGS_limit_images)) {
        break;
      }
    }
    images.swap(selected);
  }
  if (images.empty()) {
    spdlog::error("No images selected for colorization");
    return 1;
  }
  spdlog::info("Selected {} images (step={}, limit={})",
               images.size(),
               FLAGS_image_step,
               FLAGS_limit_images);
  PrintMemoryUsage();

  // loading point cloud
  spdlog::info("Loading point cloud from {} ...", FLAGS_point_cloud_filename);
  pcl::PointCloud<pcl::PointXYZRGBNormal> cloud =
      xcolor::ReadPointCloud(FLAGS_point_cloud_filename);
  PrintMemoryUsage();
  if (cloud.empty()) {
    spdlog::error("Load point cloud failed: {}", FLAGS_point_cloud_filename);
    exit(1);
  }
  spdlog::info("Load {} points.", cloud.size());

  spdlog::info("Start xcolor...");
  xcolor::FisheyeDepthOptions fisheye_depth_options;
  fisheye_depth_options.generate         = FLAGS_generate_fisheye_depths;
  fisheye_depth_options.gpu_visibility   = FLAGS_gpu_visibility;
  fisheye_depth_options.gpu_color_fusion = FLAGS_gpu_color_fusion;
  fisheye_depth_options.smooth_fusion    = FLAGS_gpu_color_smooth_fusion;
  fisheye_depth_options.scale            = FLAGS_fisheye_depth_scale;
  fisheye_depth_options.voxel_size =
      static_cast<float>(FLAGS_depth_voxel_size);
  fisheye_depth_options.max_distance =
      static_cast<float>(FLAGS_depth_max_distance);
  fisheye_depth_options.visibility_tolerance =
      static_cast<float>(FLAGS_depth_visibility_tolerance);
  fisheye_depth_options.gpu_chunk_points = FLAGS_gpu_chunk_points;
  fisheye_depth_options.output_path      = FLAGS_generated_depth_path;
  xcolor::PerformXColor(images,
                        cloud,
                        FLAGS_output_path,
                        FLAGS_max_color_candidates,
                        fisheye_depth_options);
  spdlog::info("Finish xcolor.");

  return 0;
}
