#include "xsfm_post_depth.h"

#include <colmap/geometry/rigid3.h>
#include <colmap/scene/frame.h>
#include <colmap/scene/reconstruction.h>
#include <colmap/sensor/rig.h>
#include <colmap/sensor/models.h>
#include <colmap/util/string.h>

#include <Eigen/Core>
#include <Eigen/Geometry>
#include <nlohmann/json.hpp>
#include <opencv2/opencv.hpp>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <spdlog/sinks/stdout_color_sinks.h>
#include <spdlog/spdlog.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <future>
#include <limits>
#include <map>
#include <optional>
#include <set>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace xsfm_post {
namespace fs = std::filesystem;

constexpr char kMaskExt[] = ".png";
constexpr char kDepthExt[] = ".png";
constexpr int kDefaultDepthChunkSize = 10000000;
constexpr int kDefaultMaskExpandPixels = 8;
constexpr double kRotationMatrixAtol = 1e-5;

struct FaceSpec {
  std::string name;
  Eigen::Matrix3d rotation_face_to_source;
};

struct FaceIntrinsics {
  int width = 0;
  int height = 0;
  double focal = 0.0;
  double cx = 0.0;
  double cy = 0.0;
};

struct Args {
  fs::path model_dir =
      R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\xsfm\sparse\0)";
  fs::path image_dir =
      R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\images)";
  std::optional<fs::path> mask_dir;
  fs::path output_dir =
      R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\cubemap_colmap)";
  fs::path point_cloud_path =
      R"(Z:\rick\dataset\jiuzhou\zhujiangguihuadasha\output\small_plane_refined.las)";
  std::string image_ext = ".jpg";
  int jpeg_quality = 95;
  int mask_expand_pixels = kDefaultMaskExpandPixels;
  std::string model_format = "binary";
  int limit = 0;
  int image_step = 2;
  int num_workers = 10;
  bool overwrite = false;
  bool front_only = false;
  bool generate_depths = true;
  std::string depth_mode = "dense";
  float depth_scale = 0.25f;
  float depth_voxel_size = 0.05f;
  float depth_max_distance = 30.0f;
  int gpu_chunk_points = 3000000;
};

struct RemapTable {
  cv::Mat map_x;
  cv::Mat map_y;
};

struct FaceExportPlan {
  std::string face_name;
  colmap::image_t image_id = 0;
  colmap::camera_t camera_id = 0;
  std::string output_name;
  Eigen::Matrix3d rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d translation = Eigen::Vector3d::Zero();
};

struct FaceExportResult {
  FaceExportPlan plan;
  std::vector<colmap::point3D_t> point3d_ids;
  std::vector<Eigen::Vector2d> projected_points;
  std::optional<std::string> depth_name;
  std::optional<nlohmann::json> depth_metadata;
};

struct ImageJob {
  int index = 0;
  int total = 0;
  colmap::image_t source_image_id = 0;
  std::string image_name;
  colmap::camera_t source_camera_id = 0;
  bool split_source_camera = false;
  Eigen::Matrix3d source_rotation = Eigen::Matrix3d::Identity();
  Eigen::Vector3d source_translation = Eigen::Vector3d::Zero();
  std::vector<colmap::point3D_t> point3d_ids;
  std::vector<Eigen::Vector2d> source_projected_points;
  std::vector<Eigen::Vector3d> world_xyz;
  std::vector<FaceExportPlan> face_plans;
};

struct ImageExportResult {
  int index = 0;
  std::string image_name;
  std::vector<FaceExportResult> face_results;
};

struct ExportConfig {
  fs::path image_dir;
  std::optional<fs::path> mask_dir;
  fs::path img_out;
  fs::path mask_out;
  std::optional<fs::path> depth_out;
  std::optional<fs::path> depth_colorized_out;
  std::string image_ext;
  int jpeg_quality = 95;
  std::string depth_mode;
  float depth_scale = 1.0f;
  float depth_voxel_size = 0.05f;
  float depth_max_distance = 0.0f;
  int gpu_chunk_points = 3000000;
};

struct VoxelKey {
  int64_t x = 0;
  int64_t y = 0;
  int64_t z = 0;
  bool operator==(const VoxelKey& other) const {
    return x == other.x && y == other.y && z == other.z;
  }
};

struct VoxelValue {
  Eigen::Vector3d sum = Eigen::Vector3d::Zero();
  int64_t count = 0;
};

}  // namespace xsfm_post

namespace std {
template <>
struct hash<xsfm_post::VoxelKey> {
  size_t operator()(const xsfm_post::VoxelKey& key) const {
    size_t h = std::hash<int64_t>{}(key.x);
    h ^= std::hash<int64_t>{}(key.y + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    h ^= std::hash<int64_t>{}(key.z + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2));
    return h;
  }
};
}  // namespace std

namespace xsfm_post {

std::string NormalizeRelativePath(fs::path path) {
  return path.generic_string();
}

std::string NormalizeNativeRelativePath(fs::path path) {
  std::string value = path.string();
#ifdef _WIN32
  constexpr char separator = '\\';
#else
  constexpr char separator = '/';
#endif
  std::replace(value.begin(), value.end(), '/', separator);
  std::replace(value.begin(), value.end(), '\\', separator);
  return value;
}

void InitPlaintextSpdLog() {
  auto console_sink = std::make_shared<spdlog::sinks::stdout_color_sink_mt>();
  auto logger = std::make_shared<spdlog::logger>("", console_sink);
  spdlog::set_default_logger(logger);
}

void AddCameraWithTrivialRigCompat(colmap::Reconstruction& reconstruction,
                                   colmap::Camera camera) {
  colmap::Rig rig;
  rig.SetRigId(camera.camera_id);
  rig.AddRefSensor(camera.SensorId());
  reconstruction.AddCamera(std::move(camera));
  reconstruction.AddRig(std::move(rig));
}

void AddImageWithTrivialFrameCompat(colmap::Reconstruction& reconstruction,
                                    colmap::Image image,
                                    const colmap::Rigid3d& cam_from_world) {
  const colmap::frame_t frame_id = image.ImageId();
  colmap::Frame frame;
  frame.SetFrameId(frame_id);
  frame.SetRigId(image.CameraId());
  frame.AddDataId(image.DataId());
  frame.SetRigFromWorld(cam_from_world);
  image.SetFrameId(frame.FrameId());
  reconstruction.AddFrame(std::move(frame));
  reconstruction.AddImage(std::move(image));
  reconstruction.RegisterFrame(frame_id);
}

std::string Lower(std::string value) {
  std::transform(value.begin(), value.end(), value.begin(), [](unsigned char c) {
    return static_cast<char>(std::tolower(c));
  });
  return value;
}

int EnsureEvenSize(int size) {
  size = std::max(size, 2);
  return size % 2 == 0 ? size : size + 1;
}

int RoundNearestEvenToInt(double value) {
  return static_cast<int>(std::nearbyint(value));
}

uint32_t RoundNearestEvenToUint32(double value) {
  if (value <= 0.0) {
    return 0;
  }
  return static_cast<uint32_t>(std::nearbyint(value));
}

int PixelsFromFullFov(double focal, double fov_degrees) {
  return std::max(
      RoundNearestEvenToInt(2.0 * focal * std::tan(fov_degrees * M_PI / 360.0)),
      1);
}

void ValidateRotationMatrix(const std::string& name,
                            const Eigen::Matrix3d& rotation,
                            double atol = kRotationMatrixAtol) {
  const double det = rotation.determinant();
  const double max_orthogonality_error =
      (rotation.transpose() * rotation - Eigen::Matrix3d::Identity())
          .cwiseAbs()
          .maxCoeff();
  if (!std::isfinite(det) || std::abs(det - 1.0) > atol ||
      max_orthogonality_error > atol) {
    throw std::runtime_error("Invalid rotation matrix for " + name);
  }
}

std::vector<FaceSpec> BuildFaceSpecs(bool front_only) {
  if (front_only) {
    FaceSpec front;
    front.name = "front";
    front.rotation_face_to_source = Eigen::Matrix3d::Identity();
    ValidateRotationMatrix(front.name, front.rotation_face_to_source);
    return {front};
  }

  const double sq2 = std::sqrt(2.0);
  const double sq3 = std::sqrt(3.0);
  const double sq6 = std::sqrt(6.0);
  const Eigen::Vector3d v1(1.0 / sq2, 1.0 / sq6, 1.0 / sq3);
  const Eigen::Vector3d v2(-1.0 / sq2, 1.0 / sq6, 1.0 / sq3);
  const Eigen::Vector3d v3(0.0, -std::sqrt(2.0 / 3.0), 1.0 / sq3);
  std::vector<FaceSpec> specs(3);
  specs[0].name = "face0";
  specs[0].rotation_face_to_source.col(0) = v2;
  specs[0].rotation_face_to_source.col(1) = v3;
  specs[0].rotation_face_to_source.col(2) = v1;
  specs[1].name = "face1";
  specs[1].rotation_face_to_source.col(0) = v3;
  specs[1].rotation_face_to_source.col(1) = v1;
  specs[1].rotation_face_to_source.col(2) = v2;
  specs[2].name = "face2";
  specs[2].rotation_face_to_source.col(0) = v1;
  specs[2].rotation_face_to_source.col(1) = v2;
  specs[2].rotation_face_to_source.col(2) = v3;
  for (const auto& spec : specs) {
    ValidateRotationMatrix(spec.name, spec.rotation_face_to_source);
  }
  return specs;
}

bool IsFisheyeCamera(const colmap::Camera& camera) {
  return colmap::CameraModelIsFisheye(camera.model_id) ||
         camera.ModelName().find("FISHEYE") != std::string::npos;
}

FaceIntrinsics BuildFaceIntrinsics(double shared_focal) {
  const int size = EnsureEvenSize(PixelsFromFullFov(shared_focal, 90.0f));
  return FaceIntrinsics{size, size, shared_focal, size / 2.0, size / 2.0};
}

FaceIntrinsics ScaleIntrinsics(const FaceIntrinsics& intr, float scale) {
  const int width = EnsureEvenSize(RoundNearestEvenToInt(intr.width * scale));
  const int height = EnsureEvenSize(RoundNearestEvenToInt(intr.height * scale));
  return FaceIntrinsics{width, height, intr.focal * scale, width / 2.0, height / 2.0};
}

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string key = argv[i];
    auto require_value = [&](const std::string& flag) -> std::string {
      if (i + 1 >= argc) {
        throw std::runtime_error("Missing value for " + flag);
      }
      return argv[++i];
    };
    if (key == "--model-dir" || key == "--model_dir") {
      args.model_dir = require_value(key);
    } else if (key == "--image-dir" || key == "--image_dir") {
      args.image_dir = require_value(key);
    } else if (key == "--mask-dir" || key == "--mask_dir") {
      args.mask_dir = fs::path(require_value(key));
    } else if (key == "--no-source-masks" || key == "--no_source_masks") {
      args.mask_dir.reset();
    } else if (key == "--output-dir" || key == "--output_dir") {
      args.output_dir = require_value(key);
    } else if (key == "--point-cloud-path" || key == "--point_cloud_path") {
      args.point_cloud_path = require_value(key);
    } else if (key == "--image-ext" || key == "--image_ext") {
      args.image_ext = require_value(key);
    } else if (key == "--jpeg-quality" || key == "--jpeg_quality") {
      args.jpeg_quality = std::stoi(require_value(key));
    } else if (key == "--mask-expand-pixels" || key == "--mask_expand_pixels") {
      args.mask_expand_pixels = std::stoi(require_value(key));
    } else if (key == "--model-format" || key == "--model_format") {
      args.model_format = require_value(key);
    } else if (key == "--limit") {
      args.limit = std::stoi(require_value(key));
    } else if (key == "--image-step" || key == "--image_step") {
      args.image_step = std::stoi(require_value(key));
    } else if (key == "--num-workers" || key == "--num_workers") {
      args.num_workers = std::stoi(require_value(key));
    } else if (key == "--overwrite") {
      args.overwrite = true;
    } else if (key == "--front-only" || key == "--front_only") {
      args.front_only = true;
    } else if (key == "--generate-depths" || key == "--generate_depths") {
      args.generate_depths = true;
    } else if (key == "--skip-depths" || key == "--skip_depths") {
      args.generate_depths = false;
    } else if (key == "--depth-mode" || key == "--depth_mode") {
      args.depth_mode = require_value(key);
    } else if (key == "--depth-scale" || key == "--depth_scale") {
      args.depth_scale = std::stof(require_value(key));
    } else if (key == "--depth-voxel-size" || key == "--depth_voxel_size") {
      args.depth_voxel_size = std::stof(require_value(key));
    } else if (key == "--depth-max-distance" || key == "--depth_max_distance") {
      args.depth_max_distance = std::stof(require_value(key));
    } else if (key == "--gpu-chunk-points" || key == "--gpu_chunk_points") {
      args.gpu_chunk_points = std::stoi(require_value(key));
    } else if (key == "--side-short-fov" || key == "--side-long-fov" ||
               key == "--side-angle-degrees" || key == "--mask-threshold") {
      (void)require_value(key);
    } else {
      throw std::runtime_error("Unknown argument: " + key);
    }
  }
  if (args.image_step < 1 || args.num_workers < 1 ||
      args.mask_expand_pixels < 0 || args.depth_scale <= 0.0f ||
      args.depth_voxel_size <= 0.0f || args.depth_max_distance < 0.0f ||
      args.gpu_chunk_points <= 0) {
    throw std::runtime_error("Invalid xsfm_post argument value.");
  }
  if (args.depth_mode != "dense" && args.depth_mode != "sparse") {
    throw std::runtime_error("--depth-mode must be dense or sparse.");
  }
  return args;
}

cv::Mat ReadImageAnyDepth(const fs::path& path, int flags) {
  std::ifstream input(path, std::ios::binary);
  if (!input) {
    return {};
  }
  std::vector<uchar> bytes((std::istreambuf_iterator<char>(input)),
                           std::istreambuf_iterator<char>());
  if (bytes.empty()) {
    return {};
  }
  return cv::imdecode(bytes, flags);
}

void WriteImage(const fs::path& path, const cv::Mat& image, int jpeg_quality) {
  fs::create_directories(path.parent_path());
  std::vector<int> params;
  const std::string ext = Lower(path.extension().string());
  if (ext == ".jpg" || ext == ".jpeg") {
    params = {cv::IMWRITE_JPEG_QUALITY, jpeg_quality};
  }
  std::vector<uchar> encoded;
  if (!cv::imencode(ext, image, encoded, params)) {
    throw std::runtime_error("Failed to encode image: " + path.string());
  }
  std::ofstream output(path, std::ios::binary);
  output.write(reinterpret_cast<const char*>(encoded.data()),
               static_cast<std::streamsize>(encoded.size()));
}

cv::Mat BuildMaskFromGeneratedImage(const cv::Mat& image, int expand_pixels) {
  cv::Mat black;
  if (image.channels() == 1) {
    black = image == 0;
  } else {
    std::vector<cv::Mat> channels;
    cv::split(image, channels);
    black = (channels[0] == 0) & (channels[1] == 0) & (channels[2] == 0);
  }
  if (cv::countNonZero(black) == 0) {
    return cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  }
  cv::Mat labels;
  const int count = cv::connectedComponents(black, labels, 4, CV_32S);
  std::vector<uint8_t> border(static_cast<size_t>(count), 0);
  for (int x = 0; x < labels.cols; ++x) {
    border[labels.at<int>(0, x)] = 1;
    border[labels.at<int>(labels.rows - 1, x)] = 1;
  }
  for (int y = 0; y < labels.rows; ++y) {
    border[labels.at<int>(y, 0)] = 1;
    border[labels.at<int>(y, labels.cols - 1)] = 1;
  }
  border[0] = 0;
  cv::Mat mask = cv::Mat::zeros(image.rows, image.cols, CV_8UC1);
  for (int y = 0; y < labels.rows; ++y) {
    for (int x = 0; x < labels.cols; ++x) {
      if (border[labels.at<int>(y, x)] != 0) {
        mask.at<uchar>(y, x) = 255;
      }
    }
  }
  if (expand_pixels > 0) {
    const int kernel_size = 2 * expand_pixels + 1;
    cv::Mat kernel = cv::getStructuringElement(
        cv::MORPH_ELLIPSE, cv::Size(kernel_size, kernel_size));
    cv::dilate(mask, mask, kernel);
  }
  return mask;
}

using IntrinsicsByCamera =
    std::unordered_map<colmap::camera_t, std::unordered_map<std::string, FaceIntrinsics>>;
using RemapsByCamera =
    std::unordered_map<colmap::camera_t, std::unordered_map<std::string, RemapTable>>;
using FaceCameraMap = std::map<std::pair<colmap::camera_t, std::string>, colmap::camera_t>;
using FaceMaskMap = std::map<std::pair<colmap::camera_t, std::string>, cv::Mat>;

RemapsByCamera BuildRemapTables(const colmap::Reconstruction& reconstruction,
                                const std::vector<FaceSpec>& face_specs,
                                const IntrinsicsByCamera& per_camera_intrinsics) {
  RemapsByCamera remaps;
  for (const auto& [camera_id, intr_by_face] : per_camera_intrinsics) {
    const auto& source_camera = reconstruction.Camera(camera_id);
    for (const auto& face_spec : face_specs) {
      const FaceIntrinsics& intr = intr_by_face.at(face_spec.name);
      cv::Mat map_x(intr.height, intr.width, CV_32FC1);
      cv::Mat map_y(intr.height, intr.width, CV_32FC1);
      for (int y = 0; y < intr.height; ++y) {
        for (int x = 0; x < intr.width; ++x) {
          Eigen::Vector3d ray((x - intr.cx) / intr.focal,
                              (y - intr.cy) / intr.focal,
                              1.0);
          ray.normalize();
          const Eigen::Vector3d source_ray =
              face_spec.rotation_face_to_source * ray;
          const auto source_pixel = source_camera.ImgFromCam(source_ray);
          map_x.at<float>(y, x) =
              source_pixel.has_value() ? static_cast<float>((*source_pixel).x()) : -1.0f;
          map_y.at<float>(y, x) =
              source_pixel.has_value() ? static_cast<float>((*source_pixel).y()) : -1.0f;
        }
      }
      remaps[camera_id][face_spec.name] = RemapTable{map_x, map_y};
    }
  }
  return remaps;
}

std::optional<fs::path> FindSourceMaskPath(const fs::path& mask_dir,
                                           const std::string& image_name) {
  fs::path relative(image_name);
  std::vector<fs::path> candidates = {
      mask_dir / (image_name + kMaskExt), mask_dir / relative.replace_extension(kMaskExt)};
  candidates.push_back(mask_dir / fs::path(image_name));
  const fs::path image_path(image_name);
  if (image_path.has_parent_path()) {
    const fs::path camera_name = *image_path.begin();
    candidates.push_back(mask_dir / (camera_name.string() + kMaskExt));
    candidates.push_back(mask_dir / camera_name / (std::string("mask") + kMaskExt));
  }
  for (const auto& path : candidates) {
    if (fs::exists(path)) {
      return path;
    }
  }
  return std::nullopt;
}

FaceMaskMap BuildFaceMasksFromReferenceImages(
    const colmap::Reconstruction& reconstruction,
    const std::vector<colmap::image_t>& image_ids,
    const std::vector<FaceSpec>& face_specs,
    const fs::path& image_dir,
    const RemapsByCamera& remaps,
    int expand_pixels) {
  std::map<colmap::camera_t, colmap::image_t> reference_images_by_camera;
  for (const auto image_id : image_ids) {
    const auto& image = reconstruction.Image(image_id);
    if (remaps.find(image.CameraId()) != remaps.end()) {
      reference_images_by_camera.emplace(image.CameraId(), image_id);
    }
  }

  FaceMaskMap masks;
  for (const auto& [camera_id, image_id] : reference_images_by_camera) {
    const auto& image = reconstruction.Image(image_id);
    cv::Mat source = ReadImageAnyDepth(image_dir / image.Name(),
                                       cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
    if (source.empty()) {
      throw std::runtime_error("Failed to read source image for mask generation: " +
                               (image_dir / image.Name()).string());
    }
    for (const auto& face_spec : face_specs) {
      const auto& remap = remaps.at(camera_id).at(face_spec.name);
      cv::Mat generated;
      cv::remap(source,
                generated,
                remap.map_x,
                remap.map_y,
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
      masks[{camera_id, face_spec.name}] =
          BuildMaskFromGeneratedImage(generated, expand_pixels);
    }
  }
  return masks;
}

void AppendDownsampledPoint(
    const Eigen::Vector3f& point,
    float voxel_size,
    std::unordered_map<VoxelKey, VoxelValue>& voxels) {
  if (!point.allFinite()) {
    return;
  }
  VoxelKey key;
  key.x = static_cast<int64_t>(std::floor(point.x() / voxel_size));
  key.y = static_cast<int64_t>(std::floor(point.y() / voxel_size));
  key.z = static_cast<int64_t>(std::floor(point.z() / voxel_size));
  auto& value = voxels[key];
  value.sum += point.cast<double>();
  value.count += 1;
}

std::vector<DepthWorldPoint> LoadDownsampledPointCloud(const fs::path& path,
                                                       float voxel_size) {
  std::unordered_map<VoxelKey, VoxelValue> voxels;
  const std::string ext = Lower(path.extension().string());
  if (ext == ".las" || ext == ".laz") {
    pdal::StageFactory factory;
    pdal::Stage* reader = factory.createStage("readers.las");
    pdal::Options opts;
    opts.add(pdal::Option("filename", colmap::PlatformToUTF8(path.string())));
    reader->setOptions(opts);
    pdal::PointTable table;
    reader->prepare(table);
    pdal::PointViewSet view_set = reader->execute(table);
    for (const auto& view : view_set) {
      spdlog::info("[Depth] Downsampling LAS view with {} samples", view->size());
      for (pdal::PointId i = 0; i < view->size(); ++i) {
        AppendDownsampledPoint(
            Eigen::Vector3f(view->getFieldAs<float>(pdal::Dimension::Id::X, i),
                            view->getFieldAs<float>(pdal::Dimension::Id::Y, i),
                            view->getFieldAs<float>(pdal::Dimension::Id::Z, i)),
            voxel_size,
            voxels);
      }
    }
  } else if (ext == ".pcd") {
    pcl::PointCloud<pcl::PointXYZ> cloud;
    if (pcl::io::loadPCDFile(path.string(), cloud) != 0) {
      throw std::runtime_error("Failed to load PCD: " + path.string());
    }
    spdlog::info("[Depth] Downsampling PCD with {} samples", cloud.size());
    for (const auto& p : cloud.points) {
      AppendDownsampledPoint(Eigen::Vector3f(p.x, p.y, p.z), voxel_size, voxels);
    }
  } else {
    throw std::runtime_error("Unsupported point-cloud format: " + path.extension().string());
  }

  std::vector<DepthWorldPoint> points;
  points.reserve(voxels.size());
  for (const auto& [key, value] : voxels) {
    (void)key;
    if (value.count > 0) {
      const Eigen::Vector3f point =
          (value.sum / static_cast<double>(value.count)).cast<float>();
      points.push_back({point.x(), point.y(), point.z()});
    }
  }
  return points;
}

void WriteDepthPng(const fs::path& path,
                   const std::vector<float>& depth,
                   int width,
                   int height,
                   float* min_depth,
                   float* max_depth) {
  fs::create_directories(path.parent_path());
  cv::Mat rgba(height, width, CV_8UC4, cv::Scalar(0, 0, 0, 0));
  *min_depth = std::numeric_limits<float>::infinity();
  *max_depth = 0.0f;
  bool any = false;
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      const float value = depth[static_cast<size_t>(y) * width + x];
      if (value <= 0.0f || !std::isfinite(value)) {
        continue;
      }
      any = true;
      *min_depth = std::min(*min_depth, value);
      *max_depth = std::max(*max_depth, value);
      const uint32_t mm =
          RoundNearestEvenToUint32(static_cast<double>(value * 1000.0f));
      auto& px = rgba.at<cv::Vec4b>(y, x);
      // OpenCV encodes CV_8UC4 as BGRA, while the Python script writes RGBA
      // bytes from a little-endian uint32 depth image.
      px[0] = static_cast<uint8_t>((mm >> 16) & 0xff);
      px[1] = static_cast<uint8_t>((mm >> 8) & 0xff);
      px[2] = static_cast<uint8_t>(mm & 0xff);
      px[3] = static_cast<uint8_t>((mm >> 24) & 0xff);
    }
  }
  if (!any) {
    *min_depth = 0.0f;
  }
  WriteImage(path, rgba, 95);
}

void WriteColorizedDepth(const fs::path& path,
                         const std::vector<float>& depth,
                         int width,
                         int height) {
  fs::create_directories(path.parent_path());
  cv::Mat normalized = cv::Mat::zeros(height, width, CV_8UC1);
  float min_depth = std::numeric_limits<float>::infinity();
  float max_depth = 0.0f;
  for (float value : depth) {
    if (value > 0.0f && std::isfinite(value)) {
      min_depth = std::min(min_depth, value);
      max_depth = std::max(max_depth, value);
    }
  }
  if (std::isfinite(min_depth)) {
    for (int y = 0; y < height; ++y) {
      for (int x = 0; x < width; ++x) {
        const float value = depth[static_cast<size_t>(y) * width + x];
        if (value <= 0.0f || !std::isfinite(value)) {
          continue;
        }
        normalized.at<uchar>(y, x) =
            std::abs(max_depth - min_depth) < 1e-6f
                ? 255
                : static_cast<uchar>(RoundNearestEvenToInt(
                      (value - min_depth) / (max_depth - min_depth) * 255.0));
      }
    }
  }
  cv::Mat colorized;
  cv::applyColorMap(normalized, colorized, cv::COLORMAP_JET);
  for (int y = 0; y < height; ++y) {
    for (int x = 0; x < width; ++x) {
      if (depth[static_cast<size_t>(y) * width + x] <= 0.0f) {
        colorized.at<cv::Vec3b>(y, x) = cv::Vec3b(0, 0, 0);
      }
    }
  }
  WriteImage(path, colorized, 95);
}

nlohmann::json BuildDepthMetadata(const std::string& image_name,
                                  const FaceIntrinsics& intr,
                                  const ExportConfig& config,
                                  float min_depth,
                                  float max_depth,
                                  int contributing_count,
                                  int auxiliary_count) {
  nlohmann::json metadata;
  metadata["image"] = image_name;
  metadata["width"] = intr.width;
  metadata["height"] = intr.height;
  metadata["fx"] = intr.focal;
  metadata["fy"] = intr.focal;
  metadata["cx"] = intr.cx;
  metadata["cy"] = intr.cy;
  metadata["depth_scale"] = config.depth_scale;
  metadata["voxel_size_m"] = config.depth_voxel_size;
  metadata["depth_collision_shape"] = "projected_disk_splat";
  metadata["depth_encoding"] = "rgba_uint32_le_mm";
  metadata["depth_unit"] = "mm";
  metadata["depth_scale_to_meters"] = 0.001;
  metadata["invalid_depth_value_mm"] = 0;
  metadata["contributing_voxel_count"] = contributing_count;
  metadata["min_depth_m"] = min_depth;
  metadata["max_depth_m"] = max_depth;
  if (config.depth_mode == "dense") {
    metadata["depth_output_semantics"] = "visible_splat_dense_pixels";
    metadata["filled_pixel_count"] = auxiliary_count;
  } else {
    metadata["depth_output_semantics"] = "visible_splat_center_points";
    metadata["positive_z_center_count"] = auxiliary_count;
  }
  return metadata;
}

std::pair<std::string, nlohmann::json> ExportDepthForFace(
    const CudaDepthRenderer& depth_renderer,
    const FaceExportPlan& face_plan,
    const FaceIntrinsics& intr,
    const ExportConfig& config) {
  std::string depth_name =
      NormalizeNativeRelativePath(fs::path(face_plan.output_name).replace_extension(kDepthExt));
  std::array<double, 9> rotation{};
  for (int row = 0; row < 3; ++row) {
    for (int col = 0; col < 3; ++col) {
      rotation[static_cast<size_t>(row) * 3 + col] = face_plan.rotation(row, col);
    }
  }
  const std::array<double, 3> translation = {face_plan.translation.x(),
                                             face_plan.translation.y(),
                                             face_plan.translation.z()};
  DepthRenderResult render = depth_renderer.Render(rotation,
                                                   translation,
                                                   intr,
                                                   config.depth_voxel_size,
                                                   config.gpu_chunk_points,
                                                   config.depth_max_distance,
                                                   config.depth_mode == "sparse");
  float min_depth = 0.0f;
  float max_depth = 0.0f;
  WriteDepthPng(*config.depth_out / depth_name,
                render.depth,
                render.width,
                render.height,
                &min_depth,
                &max_depth);
  if (config.depth_colorized_out.has_value()) {
    WriteColorizedDepth(*config.depth_colorized_out / depth_name,
                        render.depth,
                        render.width,
                        render.height);
  }
  return {depth_name,
          BuildDepthMetadata(face_plan.output_name,
                             intr,
                             config,
                             min_depth,
                             max_depth,
                             render.contributing_count,
                             render.auxiliary_count)};
}

int AssignFace(const Eigen::Vector3d& point_in_source,
               const std::vector<FaceSpec>& face_specs) {
  const Eigen::Vector3d normal = point_in_source.normalized();
  int best = 0;
  double best_score = -std::numeric_limits<double>::infinity();
  for (int i = 0; i < static_cast<int>(face_specs.size()); ++i) {
    const double score = normal.dot(face_specs[i].rotation_face_to_source.col(2));
    if (score > best_score) {
      best = i;
      best_score = score;
    }
  }
  return best;
}

ImageJob BuildImageJob(const colmap::Reconstruction& reconstruction,
                       colmap::image_t source_image_id,
                       int image_index,
                       int total_images,
                       const std::vector<FaceSpec>& face_specs,
                       const FaceCameraMap& face_camera_id_map,
                       const std::string& image_ext,
                       colmap::image_t* next_image_id) {
  const auto& source_image = reconstruction.Image(source_image_id);
  const auto& source_camera = reconstruction.Camera(source_image.CameraId());
  const colmap::Rigid3d pose = source_image.CamFromWorld();
  ImageJob job;
  job.index = image_index;
  job.total = total_images;
  job.source_image_id = source_image_id;
  job.image_name = source_image.Name();
  job.source_camera_id = source_image.CameraId();
  job.split_source_camera = IsFisheyeCamera(source_camera);
  job.source_rotation = pose.rotation().toRotationMatrix();
  job.source_translation = pose.translation();
  ValidateRotationMatrix("source image " + source_image.Name(), job.source_rotation);
  for (const auto& point2d : source_image.Points2D()) {
    if (point2d.HasPoint3D() && reconstruction.ExistsPoint3D(point2d.point3D_id)) {
      job.point3d_ids.push_back(point2d.point3D_id);
      job.source_projected_points.push_back(point2d.xy);
      job.world_xyz.push_back(reconstruction.Point3D(point2d.point3D_id).xyz);
    }
  }
  if (job.split_source_camera) {
    const fs::path image_path(source_image.Name());
    const std::string rel_path = image_path.parent_path().filename().string();
    const std::string stem = image_path.stem().string();
    for (const auto& face_spec : face_specs) {
      FaceExportPlan plan;
      plan.face_name = face_spec.name;
      plan.image_id = (*next_image_id)++;
      plan.camera_id = face_camera_id_map.at({source_image.CameraId(), face_spec.name});
      plan.output_name =
          NormalizeRelativePath(fs::path(rel_path + "_" + face_spec.name) /
                                (stem + image_ext));
      plan.rotation = face_spec.rotation_face_to_source.transpose() * job.source_rotation;
      plan.translation = face_spec.rotation_face_to_source.transpose() * job.source_translation;
      ValidateRotationMatrix(source_image.Name() + " " + face_spec.name, plan.rotation);
      job.face_plans.push_back(plan);
    }
  } else {
    FaceExportPlan plan;
    plan.face_name = "original";
    plan.image_id = (*next_image_id)++;
    plan.camera_id = face_camera_id_map.at({source_image.CameraId(), "original"});
    plan.output_name = source_image.Name();
    plan.rotation = job.source_rotation;
    plan.translation = job.source_translation;
    job.face_plans.push_back(plan);
  }
  return job;
}

ImageExportResult ProcessImageJob(
    const ImageJob& job,
    const std::vector<FaceSpec>& face_specs,
    const FaceMaskMap& face_masks,
    const IntrinsicsByCamera& per_camera_intrinsics,
    const IntrinsicsByCamera& depth_intrinsics,
    const RemapsByCamera& remaps,
    const CudaDepthRenderer* depth_renderer,
    const ExportConfig& config) {
  cv::Mat source_pixels = ReadImageAnyDepth(config.image_dir / job.image_name,
                                            cv::IMREAD_COLOR | cv::IMREAD_IGNORE_ORIENTATION);
  if (source_pixels.empty()) {
    throw std::runtime_error("Failed to read source image: " +
                             (config.image_dir / job.image_name).string());
  }
  std::optional<cv::Mat> source_mask;
  if (config.mask_dir.has_value()) {
    const auto source_mask_path = FindSourceMaskPath(*config.mask_dir, job.image_name);
    if (source_mask_path.has_value()) {
      source_mask = ReadImageAnyDepth(*source_mask_path,
                                      cv::IMREAD_GRAYSCALE |
                                          cv::IMREAD_IGNORE_ORIENTATION);
      if (source_mask->empty()) {
        throw std::runtime_error("Failed to read source mask: " +
                                 source_mask_path->string());
      }
    } else {
      spdlog::warn("[Mask] Missing source mask for {}; using full-valid mask",
                   job.image_name);
      source_mask = cv::Mat(source_pixels.rows, source_pixels.cols, CV_8UC1, cv::Scalar(255));
    }
  }

  std::map<std::string, std::pair<std::vector<colmap::point3D_t>,
                                  std::vector<Eigen::Vector2d>>>
      projected_by_face;
  if (!job.split_source_camera) {
    projected_by_face["original"] = {job.point3d_ids, job.source_projected_points};
  } else {
    for (const auto& face_spec : face_specs) {
      projected_by_face[face_spec.name] = {};
    }
    for (size_t i = 0; i < job.world_xyz.size(); ++i) {
      const Eigen::Vector3d point_in_source =
          job.source_rotation * job.world_xyz[i] + job.source_translation;
      const int face_index = AssignFace(point_in_source, face_specs);
      const auto& face_spec = face_specs[face_index];
      const FaceIntrinsics& intr =
          per_camera_intrinsics.at(job.source_camera_id).at(face_spec.name);
      const Eigen::Vector3d point_in_face =
          face_spec.rotation_face_to_source.transpose() * point_in_source;
      if (point_in_face.z() <= 0.0) {
        continue;
      }
      const Eigen::Vector2d projected(intr.focal * point_in_face.x() / point_in_face.z() +
                                          intr.cx,
                                      intr.focal * point_in_face.y() / point_in_face.z() +
                                          intr.cy);
      if (projected.x() >= 0.0 && projected.x() < intr.width &&
          projected.y() >= 0.0 && projected.y() < intr.height) {
        projected_by_face[face_spec.name].first.push_back(job.point3d_ids[i]);
        projected_by_face[face_spec.name].second.push_back(projected);
      }
    }
  }

  ImageExportResult result;
  result.index = job.index;
  result.image_name = job.image_name;
  for (const auto& plan : job.face_plans) {
    cv::Mat resampled_image;
    cv::Mat resampled_mask;
    if (job.split_source_camera) {
      const auto& remap = remaps.at(job.source_camera_id).at(plan.face_name);
      cv::remap(source_pixels,
                resampled_image,
                remap.map_x,
                remap.map_y,
                cv::INTER_LINEAR,
                cv::BORDER_CONSTANT,
                cv::Scalar(0, 0, 0));
      if (source_mask.has_value()) {
        cv::remap(*source_mask,
                  resampled_mask,
                  remap.map_x,
                  remap.map_y,
                  cv::INTER_NEAREST,
                  cv::BORDER_CONSTANT,
                  cv::Scalar(0));
      } else {
        resampled_mask = face_masks.at({job.source_camera_id, plan.face_name});
      }
      WriteImage(config.img_out / plan.output_name,
                 resampled_image,
                 config.jpeg_quality);
    } else {
      resampled_mask = source_mask.value_or(
          cv::Mat(source_pixels.rows, source_pixels.cols, CV_8UC1, cv::Scalar(0)));
      fs::create_directories((config.img_out / plan.output_name).parent_path());
      fs::copy_file(config.image_dir / job.image_name,
                    config.img_out / plan.output_name,
                    fs::copy_options::overwrite_existing);
    }
    const std::string mask_name =
        NormalizeRelativePath(fs::path(plan.output_name).replace_extension(kMaskExt));
    WriteImage(config.mask_out / mask_name, resampled_mask, config.jpeg_quality);

    FaceExportResult face_result;
    face_result.plan = plan;
    face_result.point3d_ids = projected_by_face[plan.face_name].first;
    face_result.projected_points = projected_by_face[plan.face_name].second;
    if (depth_renderer != nullptr && config.depth_out.has_value()) {
      if (!job.split_source_camera) {
        throw std::runtime_error(
            "Depth export is not supported for passthrough non-fisheye cameras.");
      }
      auto [depth_name, metadata] = ExportDepthForFace(
          *depth_renderer,
          plan,
          depth_intrinsics.at(job.source_camera_id).at(plan.face_name),
          config);
      face_result.depth_name = depth_name;
      face_result.depth_metadata = metadata;
    }
    result.face_results.push_back(std::move(face_result));
  }
  spdlog::info("[Image] Finished {}/{}: {}", job.index, job.total, job.image_name);
  return result;
}

std::vector<colmap::Point2D> BuildLinkedPoints2D(
    const std::vector<Eigen::Vector2d>& projected,
    const std::vector<colmap::point3D_t>& point3d_ids) {
  if (projected.size() != point3d_ids.size()) {
    throw std::runtime_error("Projected point count and Point3D id count mismatch.");
  }
  std::vector<colmap::Point2D> points(projected.size());
  for (size_t i = 0; i < projected.size(); ++i) {
    points[i].xy = projected[i];
    points[i].point3D_id = point3d_ids[i];
  }
  return points;
}

void WriteJson(const fs::path& path, const nlohmann::json& content) {
  fs::create_directories(path.parent_path());
  std::ofstream output(path);
  output << content.dump(2) << '\n';
}

int Run(int argc, char** argv) {
  InitPlaintextSpdLog();
  Args args = ParseArgs(argc, argv);
  if (args.generate_depths && !HasCudaDevice()) {
    throw std::runtime_error(
        "CUDA depth rendering was requested but no CUDA device is available.");
  }
  if (args.mask_dir.has_value() && !fs::exists(*args.mask_dir)) {
    throw std::runtime_error("Source mask directory not found: " +
                             args.mask_dir->string());
  }
  if (args.num_workers > 1) {
    cv::setNumThreads(1);
  }
  if (args.overwrite && fs::exists(args.output_dir)) {
    fs::remove_all(args.output_dir);
  }

  colmap::Reconstruction reconstruction;
  reconstruction.Read(args.model_dir.string());
  const std::vector<FaceSpec> face_specs = BuildFaceSpecs(args.front_only);
  spdlog::info("[Setup] Perspective views per fisheye image: {} ({})",
               face_specs.size(),
               args.front_only ? "front-only" : "three-face coverage");

  std::set<colmap::camera_t> fisheye_camera_ids;
  std::set<colmap::camera_t> non_fisheye_camera_ids;
  for (const auto& [camera_id, camera] : reconstruction.Cameras()) {
    if (IsFisheyeCamera(camera)) {
      fisheye_camera_ids.insert(camera_id);
    } else {
      non_fisheye_camera_ids.insert(camera_id);
    }
  }
  if (args.generate_depths && !non_fisheye_camera_ids.empty()) {
    throw std::runtime_error(
        "Depth export is only supported when all cameras are fisheye-split. "
        "Use --skip-depths to keep non-fisheye cameras unchanged.");
  }

  IntrinsicsByCamera per_camera_intrinsics;
  IntrinsicsByCamera depth_intrinsics;
  for (const auto camera_id : fisheye_camera_ids) {
    const double focal = reconstruction.Camera(camera_id).MeanFocalLength();
    for (const auto& face_spec : face_specs) {
      per_camera_intrinsics[camera_id][face_spec.name] = BuildFaceIntrinsics(focal);
      depth_intrinsics[camera_id][face_spec.name] =
          ScaleIntrinsics(per_camera_intrinsics[camera_id][face_spec.name],
                          args.depth_scale);
    }
  }

  spdlog::info("[Setup] Building remap tables");
  RemapsByCamera remaps =
      BuildRemapTables(reconstruction, face_specs, per_camera_intrinsics);

  std::vector<colmap::image_t> image_ids;
  image_ids.reserve(reconstruction.Images().size());
  for (const auto& [image_id, image] : reconstruction.Images()) {
    (void)image;
    image_ids.push_back(image_id);
  }
  std::sort(image_ids.begin(), image_ids.end(), [&](auto a, auto b) {
    return reconstruction.Image(a).Name() < reconstruction.Image(b).Name();
  });
  if (args.image_step > 1) {
    std::vector<colmap::image_t> stepped;
    for (size_t i = 0; i < image_ids.size(); i += args.image_step) {
      stepped.push_back(image_ids[i]);
    }
    image_ids.swap(stepped);
  }
  if (args.limit > 0 && static_cast<size_t>(args.limit) < image_ids.size()) {
    image_ids.resize(args.limit);
  }

  FaceMaskMap face_masks;
  if (args.mask_dir.has_value()) {
    spdlog::info("[Setup] Using per-image source masks: {}", args.mask_dir->string());
  } else {
    spdlog::info("[Setup] Building shared masks from reference face images");
    face_masks = BuildFaceMasksFromReferenceImages(reconstruction,
                                                   image_ids,
                                                   face_specs,
                                                   args.image_dir,
                                                   remaps,
                                                   args.mask_expand_pixels);
  }

  std::vector<DepthWorldPoint> depth_points_world;
  std::unique_ptr<CudaDepthRenderer> depth_renderer;
  if (args.generate_depths) {
    if (!fs::exists(args.point_cloud_path)) {
      throw std::runtime_error("Point cloud not found: " +
                               args.point_cloud_path.string());
    }
    spdlog::info("Loading and downsampling point cloud: {}",
                 args.point_cloud_path.string());
    depth_points_world =
        LoadDownsampledPointCloud(args.point_cloud_path, args.depth_voxel_size);
    spdlog::info("Depth points after downsampling: {}", depth_points_world.size());
    spdlog::info("Uploading depth points to CUDA once");
    depth_renderer = std::make_unique<CudaDepthRenderer>(depth_points_world);
    depth_points_world.clear();
    depth_points_world.shrink_to_fit();
  }

  const fs::path img_out = args.output_dir / "images";
  const fs::path mask_out = args.output_dir / "masks";
  fs::create_directories(img_out);
  fs::create_directories(mask_out);
  std::optional<fs::path> depth_out;
  std::optional<fs::path> depth_colorized_out;
  if (args.generate_depths) {
    depth_out = args.output_dir / "depths";
    depth_colorized_out = args.output_dir / "depth_colorized";
    fs::create_directories(*depth_out);
    fs::create_directories(*depth_colorized_out);
  }

  colmap::Reconstruction converted;
  FaceCameraMap face_camera_id_map;
  colmap::camera_t next_camera_id = 1;
  for (const auto& [source_camera_id, source_camera] : reconstruction.Cameras()) {
    if (fisheye_camera_ids.count(source_camera_id) != 0) {
      for (const auto& face_spec : face_specs) {
        const auto& intr = per_camera_intrinsics.at(source_camera_id).at(face_spec.name);
        colmap::Camera camera;
        camera.camera_id = next_camera_id;
        camera.model_id = colmap::CameraModelId::kPinhole;
        camera.width = intr.width;
        camera.height = intr.height;
        camera.params = {intr.focal, intr.focal, intr.cx, intr.cy};
        AddCameraWithTrivialRigCompat(converted, std::move(camera));
        face_camera_id_map[{source_camera_id, face_spec.name}] = next_camera_id++;
      }
    } else {
      colmap::Camera camera = source_camera;
      camera.camera_id = next_camera_id;
      AddCameraWithTrivialRigCompat(converted, std::move(camera));
      face_camera_id_map[{source_camera_id, "original"}] = next_camera_id++;
    }
  }

  spdlog::info("[Image] Selected {}/{} source images (sorted by name, step={}, workers={}, depths={})",
               image_ids.size(),
               reconstruction.Images().size(),
               args.image_step,
               args.num_workers,
               args.generate_depths ? "cuda/" + args.depth_mode : "skipped");

  ExportConfig config;
  config.image_dir = args.image_dir;
  config.mask_dir = args.mask_dir;
  config.img_out = img_out;
  config.mask_out = mask_out;
  config.depth_out = depth_out;
  config.depth_colorized_out = depth_colorized_out;
  config.image_ext = args.image_ext;
  config.jpeg_quality = args.jpeg_quality;
  config.depth_mode = args.depth_mode;
  config.depth_scale = args.depth_scale;
  config.depth_voxel_size = args.depth_voxel_size;
  config.depth_max_distance =
      args.depth_max_distance == 0.0f ? -1.0f : args.depth_max_distance;
  config.gpu_chunk_points = args.gpu_chunk_points;

  std::vector<ImageJob> jobs;
  jobs.reserve(image_ids.size());
  colmap::image_t next_image_id = 1;
  for (size_t i = 0; i < image_ids.size(); ++i) {
    jobs.push_back(BuildImageJob(reconstruction,
                                 image_ids[i],
                                 static_cast<int>(i + 1),
                                 static_cast<int>(image_ids.size()),
                                 face_specs,
                                 face_camera_id_map,
                                 args.image_ext,
                                 &next_image_id));
  }

  std::vector<ImageExportResult> image_results(jobs.size());
  if (args.num_workers <= 1 || jobs.size() <= 1) {
    for (size_t i = 0; i < jobs.size(); ++i) {
      image_results[i] = ProcessImageJob(jobs[i],
                                         face_specs,
                                         face_masks,
                                         per_camera_intrinsics,
                                         depth_intrinsics,
                                         remaps,
                                         depth_renderer.get(),
                                         config);
    }
  } else {
    size_t next_job = 0;
    while (next_job < jobs.size()) {
      std::vector<std::future<ImageExportResult>> futures;
      for (int worker = 0; worker < args.num_workers && next_job < jobs.size();
           ++worker, ++next_job) {
        futures.push_back(std::async(std::launch::async,
                                     ProcessImageJob,
                                     jobs[next_job],
                                     std::cref(face_specs),
                                     std::cref(face_masks),
                                     std::cref(per_camera_intrinsics),
                                     std::cref(depth_intrinsics),
                                     std::cref(remaps),
                                     depth_renderer.get(),
                                     std::cref(config)));
      }
      for (auto& future : futures) {
        ImageExportResult result = future.get();
        image_results[result.index - 1] = std::move(result);
      }
    }
  }

  std::unordered_map<colmap::point3D_t, std::vector<colmap::TrackElement>>
      tracks_by_point3d;
  nlohmann::json depth_metadata_by_name = nlohmann::json::object();
  for (const auto& image_result : image_results) {
    for (const auto& face_result : image_result.face_results) {
      colmap::Image image;
      image.SetName(face_result.plan.output_name);
      image.SetCameraId(face_result.plan.camera_id);
      image.SetImageId(face_result.plan.image_id);
      image.SetPoints2D(BuildLinkedPoints2D(face_result.projected_points,
                                            face_result.point3d_ids));
      AddImageWithTrivialFrameCompat(
          converted,
          image,
          colmap::Rigid3d(Eigen::Quaterniond(face_result.plan.rotation).normalized(),
                          face_result.plan.translation));
      for (size_t point_index = 0; point_index < face_result.point3d_ids.size();
           ++point_index) {
        tracks_by_point3d[face_result.point3d_ids[point_index]].emplace_back(
            face_result.plan.image_id,
            static_cast<colmap::point2D_t>(point_index));
      }
      if (face_result.depth_name.has_value() &&
          face_result.depth_metadata.has_value()) {
        depth_metadata_by_name[*face_result.depth_name] = *face_result.depth_metadata;
      }
    }
  }

  for (const auto& [point3d_id, track_elements] : tracks_by_point3d) {
    if (!reconstruction.ExistsPoint3D(point3d_id)) {
      continue;
    }
    const auto& src_point = reconstruction.Point3D(point3d_id);
    colmap::Track track;
    for (const auto& element : track_elements) {
      track.AddElement(element);
    }
    colmap::Point3D point;
    point.xyz = src_point.xyz;
    point.color = src_point.color;
    point.error = src_point.error;
    point.track = std::move(track);
    converted.AddPoint3D(point3d_id, std::move(point));
  }

  const fs::path out_model = args.output_dir / "sparse";
  fs::create_directories(out_model);
  if (args.model_format == "binary" || args.model_format == "both") {
    converted.WriteBinary(out_model.string());
  }
  if (args.model_format == "text" || args.model_format == "both") {
    converted.WriteText(out_model.string());
  }
  if (depth_out.has_value()) {
    WriteJson(args.output_dir / "depth_intrinsics.json", depth_metadata_by_name);
  }
  spdlog::info("Finished. Images: {}, Masks: {}, Depths: {}",
               img_out.string(),
               mask_out.string(),
               depth_out.has_value() ? depth_out->string() : "skipped");
  return 0;
}

}  // namespace xsfm_post

int main(int argc, char** argv) {
  try {
    return xsfm_post::Run(argc, argv);
  } catch (const std::exception& e) {
    spdlog::error("{}", e.what());
    return 1;
  }
}
