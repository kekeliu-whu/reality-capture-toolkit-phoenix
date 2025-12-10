
#include <colmap/scene/database.h>
#include <colmap/scene/image.h>
#include <colmap/scene/reconstruction.h>
#include <fstream>
#include <spdlog/spdlog.h>

#include "io/colmap_io.h"

namespace xcolor {

template <typename T>
T NativeToLittleEndian(const T x) {
  return x;
}

template <typename T>
T ReadBinaryLittleEndian(std::istream* stream) {
  T data_little_endian;
  stream->read(reinterpret_cast<char*>(&data_little_endian), sizeof(T));
  return data_little_endian;
}

template <typename T>
void ReadBinaryLittleEndian(std::istream* stream, std::vector<T>* data) {
  for (size_t i = 0; i < data->size(); ++i) {
    (*data)[i] = ReadBinaryLittleEndian<T>(stream);
  }
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const T& data) {
  const T data_little_endian = NativeToLittleEndian(data);
  stream->write(reinterpret_cast<const char*>(&data_little_endian), sizeof(T));
}

template <typename T>
void WriteBinaryLittleEndian(std::ostream* stream, const std::vector<T>& data) {
  for (const auto& elem : data) {
    WriteBinaryLittleEndian<T>(stream, elem);
  }
}

void ReadCamerasBinary(const std::string& filename, std::vector<colmap::Camera>& cameras) {
  std::ifstream stream(filename, std::ios::binary);
  using namespace colmap;

  const size_t num_cameras = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_cameras; ++i) {
    struct Camera camera;
    camera.camera_id = ReadBinaryLittleEndian<camera_t>(&stream);
    camera.model_id  = static_cast<CameraModelId>(ReadBinaryLittleEndian<int>(&stream));
    camera.width     = ReadBinaryLittleEndian<uint64_t>(&stream);
    camera.height    = ReadBinaryLittleEndian<uint64_t>(&stream);
    camera.params.resize(CameraModelNumParams(camera.model_id), 0.);
    ReadBinaryLittleEndian<double>(&stream, &camera.params);
    cameras.push_back(std::move(camera));
  }
}

// msvc seems to have an internal bug in ifstream::read
// which causes an error with errno == 2 midway
//
// example:
//
// colmap::Reconstruction reconstruction;
// reconstruction.ReadBinary(FLAGS_sfm_result_path);
// auto images = reconstruction.Images();
// DLOG(INFO) << images.size();
//
inline void ReadImagesBinary(const std::string& filename, std::vector<colmap::Image>& images) {
  std::ifstream stream(filename, std::ios::binary);
  if (!stream.good()) { spdlog::error("Failed to open {}", filename); exit(1); }
  std::vector<Eigen::Vector2d> points2D;
  std::vector<colmap::point3D_t> point3D_ids;

  const size_t num_reg_images = ReadBinaryLittleEndian<uint64_t>(&stream);
  for (size_t i = 0; i < num_reg_images; ++i) {
    colmap::Image image;

    image.SetImageId(ReadBinaryLittleEndian<colmap::image_t>(&stream));

    colmap::Rigid3d cam_from_world;
    cam_from_world.rotation.w()    = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.x()    = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.y()    = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.rotation.z()    = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.x() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.y() = ReadBinaryLittleEndian<double>(&stream);
    cam_from_world.translation.z() = ReadBinaryLittleEndian<double>(&stream);
    image.SetCamFromWorld(cam_from_world);

    image.SetCameraId(ReadBinaryLittleEndian<colmap::camera_t>(&stream));

    char name_char;
    do {
      stream.read(&name_char, 1);
      if (name_char != '\0') {
        image.Name() += name_char;
      }
    } while (name_char != '\0');

    const size_t num_points2D = ReadBinaryLittleEndian<uint64_t>(&stream);

    points2D.clear();
    points2D.reserve(num_points2D);
    point3D_ids.clear();
    point3D_ids.reserve(num_points2D);
    for (size_t j = 0; j < num_points2D; ++j) {
      const double x = ReadBinaryLittleEndian<double>(&stream);
      const double y = ReadBinaryLittleEndian<double>(&stream);
      points2D.emplace_back(x, y);
      point3D_ids.push_back(ReadBinaryLittleEndian<colmap::point3D_t>(&stream));
    }

    image.SetPoints2D(points2D);

    for (colmap::point2D_t point2D_idx = 0; point2D_idx < image.NumPoints2D(); ++point2D_idx) {
      if (point3D_ids[point2D_idx] != colmap::kInvalidPoint3DId) {
        image.SetPoint3DForPoint2D(point2D_idx, point3D_ids[point2D_idx]);
      }
    }

    images.push_back(image);
  }
}

void ReadImages(const std::string& sfm_path, const std::string& images_path, std::vector<Image>& images) {
  spdlog::info("Loading image poses from {} ...", sfm_path + "/images.bin");
  std::vector<colmap::Image> raw_images;
  ReadImagesBinary(sfm_path + "/images.bin", raw_images);
  spdlog::info("Load {} image poses.", raw_images.size());

  spdlog::info("Loading cameras from {} ...", sfm_path + "/cameras.bin");
  std::vector<colmap::Camera> raw_cameras;
  ReadCamerasBinary(sfm_path + "/cameras.bin", raw_cameras);
  spdlog::info("Load {} cameras.", raw_cameras.size());

  spdlog::info("Loading images from {} ...", images_path);
  std::unordered_map<colmap::camera_t, colmap::Camera> camera_id2camera;
  for (auto& e : raw_cameras) {
    camera_id2camera[e.camera_id] = e;
  }
  images.clear();
  for (int i = 0; i < raw_images.size(); i++) {
    auto raw_image     = raw_images[i];
    cv::Mat img_opencv = cv::imread(images_path + "/" + raw_image.Name());
    if (img_opencv.empty()) { spdlog::error("Check failed"); exit(1); }
    if (img_opencv.channels() != 3) { spdlog::error("Check failed"); exit(1); }

    Image image;
    image.image  = img_opencv;
    image.pose   = raw_image.CamFromWorld();
    image.camera = camera_id2camera.at(raw_image.CameraId());
    image.name   = raw_image.Name();
    images.push_back(image);
  }
  spdlog::info("Load {} images.", images.size());
}

void WriteCamerasBinary(const std::string& filename, const std::vector<colmap::Camera>& cameras) {
  std::ofstream stream(filename, std::ios::binary);
  if (!stream.good()) {
    spdlog::error("Failed to open file: {}", filename);
    exit(1);
  }

  WriteBinaryLittleEndian<uint64_t>(&stream, cameras.size());

  for (const auto& camera : cameras) {
    WriteBinaryLittleEndian<colmap::camera_t>(&stream, camera.camera_id);
    WriteBinaryLittleEndian<int>(&stream, static_cast<int>(camera.model_id));
    WriteBinaryLittleEndian<uint64_t>(&stream, camera.width);
    WriteBinaryLittleEndian<uint64_t>(&stream, camera.height);
    for (const double param : camera.params) {
      WriteBinaryLittleEndian<double>(&stream, param);
    }
  }
}

void WriteImagesBinary(const std::string& filename, const std::vector<colmap::Image>& images) {
  std::ofstream stream(filename, std::ios::binary);
  if (!stream.good()) { spdlog::error("Check failed"); exit(1); }

  WriteBinaryLittleEndian<uint64_t>(&stream, images.size());

  for (const auto& image : images) {
    WriteBinaryLittleEndian<colmap::image_t>(&stream, image.ImageId());

    const colmap::Rigid3d& cam_from_world = image.CamFromWorld();
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.w());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.x());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.y());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.rotation.z());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.x());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.y());
    WriteBinaryLittleEndian<double>(&stream, cam_from_world.translation.z());

    WriteBinaryLittleEndian<colmap::camera_t>(&stream, image.CameraId());

    const std::string name = image.Name() + '\0';
    stream.write(name.c_str(), name.size());

    WriteBinaryLittleEndian<uint64_t>(&stream, image.NumPoints2D());
    for (const colmap::Point2D& point2D : image.Points2D()) {
      WriteBinaryLittleEndian<double>(&stream, point2D.xy(0));
      WriteBinaryLittleEndian<double>(&stream, point2D.xy(1));
      WriteBinaryLittleEndian<colmap::point3D_t>(&stream, point2D.point3D_id);
    }
  }
}

void WritePoints3DBinary(const std::string& filename, const std::vector<colmap::Point3D>& points3D) {
  std::ofstream stream(filename, std::ios::binary);
  if (!stream.good()) { spdlog::error("Check failed"); exit(1); }

  WriteBinaryLittleEndian<uint64_t>(&stream, points3D.size());
  stream.flush();

  for (int i = 0; i < points3D.size(); i++) {
    const colmap::Point3D& point3D = points3D[i];

    WriteBinaryLittleEndian<colmap::point3D_t>(&stream, i);
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(0));
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(1));
    WriteBinaryLittleEndian<double>(&stream, point3D.xyz(2));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(0));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(1));
    WriteBinaryLittleEndian<uint8_t>(&stream, point3D.color(2));
    WriteBinaryLittleEndian<double>(&stream, point3D.error);

    WriteBinaryLittleEndian<uint64_t>(&stream, point3D.track.Length());
    for (const colmap::TrackElement& track_el : point3D.track.Elements()) {
      WriteBinaryLittleEndian<colmap::image_t>(&stream, track_el.image_id);
      WriteBinaryLittleEndian<colmap::point2D_t>(&stream, track_el.point2D_idx);
    }
  }
}

}  // namespace xcolor
