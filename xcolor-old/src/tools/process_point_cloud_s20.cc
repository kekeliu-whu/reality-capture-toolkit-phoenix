#include <glog/logging.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <proj.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <fstream>
#include <nlohmann/json.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>

DEFINE_string(las_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/colorized.las", "Input project path");
DEFINE_string(initial_pose_filename, "D:/ProjectX/project-3d/data/sfm/area2-1/sfm/images/ImgPose.txt", "Initial pose filename");
DEFINE_bool(output_full, false, "Output full point cloud");

static constexpr double kDownsampleVoxelSize    = 0.05;
static constexpr int kNearestNeighbors          = 15;
static constexpr int kSmoothMaxNearestNeighbors = 100;
static constexpr double kSmoothMaxSearchRadius  = 0.3;
static constexpr double kSmoothSigmaD           = 0.05;
static constexpr double kSmoothSigmaN           = 0.05;

class LocalENUTransformer {
 public:
  LocalENUTransformer(const Eigen::Vector2d &offset, const std::string &proj_str) {
    offset_   = offset;
    proj_str_ = proj_str;
    ctx_      = proj_context_create();
    origin_   = ComputeOriginLonLat(ctx_, offset_, proj_str_);

    std::ostringstream oss;
    oss << std::fixed << std::setprecision(9) << "+proj=tmerc +lat_0=" << origin_.y() << " +lon_0=" << origin_.x()
        << " +x_0=0 +y_0=0 +datum=WGS84 +units=m +no_defs";
    std::string target_enu_proj_str = oss.str();
    transformer_                    = proj_create_crs_to_crs(ctx_, proj_str_.c_str(), target_enu_proj_str.c_str(), nullptr);
    DCHECK(transformer_) << "Failed to create transformer with proj string: " << proj_str_ << " to " << target_enu_proj_str;
  }

  ~LocalENUTransformer() {
    if (transformer_) {
      proj_destroy(transformer_);
      transformer_ = nullptr;
    }
    if (ctx_) {
      proj_context_destroy(ctx_);
      ctx_ = nullptr;
    }
  }

  Eigen::Vector2d Transform(const Eigen::Vector2d &coord) const {
    PJ_COORD coord_source = proj_coord(coord.x(), coord.y(), 0, 0);
    PJ_COORD coord_enu    = proj_trans(transformer_, PJ_FWD, coord_source);

    Eigen::Vector2d result = Eigen::Vector2d::Zero();
    if (coord_enu.xy.x != HUGE_VAL && coord_enu.xy.y != HUGE_VAL) {
      result.x() = coord_enu.enu.e;
      result.y() = coord_enu.enu.n;
    }
    return result;
  }

  Eigen::Vector2d getOriginLonLat() const { return origin_; }

  Eigen::Vector2d getOffset() const { return offset_; }

  double GetMeridianConvergence() const {
    double lon = origin_.x();
    double lat = origin_.y();

    PJ_COORD coord_ll = proj_coord(lon / 180 * M_PI, lat / 180 * M_PI, 0, 0);

    PJ *proj = proj_create(ctx_, proj_str_.c_str());

    auto factors = proj_factors(proj, coord_ll);
    proj_destroy(proj);

    DLOG(INFO) << "Meridian convergence at lon: " << lon << ", lat: " << lat << " is " << factors.meridian_convergence * 180.0 / M_PI;
    DLOG(INFO) << "Mercator scale at lon: " << lon << ", lat: " << lat << " is " << factors.meridional_scale << ", " << factors.parallel_scale;

    return factors.meridian_convergence * 180.0 / M_PI;
  }

 private:
  Eigen::Vector2d ComputeOriginLonLat(PJ_CONTEXT *ctx, const Eigen::Vector2d &offset, const std::string &proj_str) const {
    PJ *transformer = proj_create_crs_to_crs(ctx, proj_str.c_str(), "EPSG:4326", nullptr);
    DCHECK(transformer) << "Failed to create transformer with proj string: " << proj_str << " to EPSG:4326";

    Eigen::Vector2d result = Eigen::Vector2d::Zero();
    if (transformer) {
      PJ_COORD coord_source = proj_coord(offset.x(), offset.y(), 0, 0);
      PJ_COORD coord_ll     = proj_trans(transformer, PJ_FWD, coord_source);
      if (coord_ll.lp.lam != HUGE_VAL && coord_ll.lp.phi != HUGE_VAL) {
        result.x() = coord_ll.xy.y;
        result.y() = coord_ll.xy.x;
      }
      proj_destroy(transformer);
    }
    return result;
  }

 private:
  std::string proj_str_;
  Eigen::Vector2d offset_;  // east,north
  Eigen::Vector2d origin_;  // longitude,latitude
  PJ_CONTEXT *ctx_;
  PJ *transformer_;
};

class VoxelLoc {
 public:
  int64_t x, y, z;

  VoxelLoc(int64_t vx = 0, int64_t vy = 0, int64_t vz = 0) : x(vx), y(vy), z(vz) {}

  bool operator==(const VoxelLoc &other) const { return (x == other.x && y == other.y && z == other.z); }
};

// Hash value
namespace std {
template <>
struct hash<VoxelLoc> {
  int64_t operator()(const VoxelLoc &s) const {
    using std::hash;
    using std::size_t;
    constexpr uint64_t HASH_P = 116101;
    constexpr uint64_t MAX_N  = 10000000000UL;
    return ((((s.z) * HASH_P) % MAX_N + (s.y)) * HASH_P) % MAX_N + (s.x);
  }
};
}  // namespace std

struct M_POINT {
  Eigen::Vector3d center;
  int count = 0;
};

template <typename PointType>
void DownsamplePointCloudInternal(const pcl::PointCloud<PointType> &cloud_in, pcl::PointCloud<PointType> &cloud_out, double voxel_size) {
  if (voxel_size < 0.01) {
    return;
  }

  std::unordered_map<VoxelLoc, M_POINT> feat_map;

  for (int i = 0; i < cloud_in.size(); i++) {
    Eigen::Vector3d p_c = cloud_in[i].getVector3fMap().template cast<double>();
    int loc_xyz[3];
    for (int j = 0; j < 3; j++) {
      loc_xyz[j] = p_c[j] / voxel_size;
      if (loc_xyz[j] < 0) {
        loc_xyz[j] -= 1;
      }
    }

    VoxelLoc position(loc_xyz[0], loc_xyz[1], loc_xyz[2]);
    auto iter = feat_map.find(position);
    if (iter != feat_map.end()) {
      iter->second.center += p_c;
      iter->second.count++;
    } else {
      M_POINT p;
      p.center           = p_c;
      p.count            = 1;
      feat_map[position] = p;
    }
  }

  cloud_out.clear();
  cloud_out.resize(feat_map.size());

  int i = 0;
  for (auto iter = feat_map.begin(); iter != feat_map.end(); ++iter) {
    cloud_out[i].getVector3fMap() = iter->second.center.cast<float>() / iter->second.count;
    i++;
  }
}

void SaveLasOffset(const std::string &filename, double offset_x, double offset_y, double lon, double lat) {
  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    LOG(ERROR) << "Failed to open file: " << filename;
    return;
  }
  ofs << "offset_x,offset_y,lon,lat" << std::endl;
  ofs << std::fixed << std::setprecision(9) << offset_x << "," << offset_y << "," << lon << "," << lat << std::endl;
}

void SaveLasOffsetJson(const std::string &filename, Eigen::Vector2d &offset, const std::string &proj4_str, const Eigen::Vector2d &origin_lonlat) {
  nlohmann::json j;
  j["offset_x"]     = offset.x();
  j["offset_y"]     = offset.y();
  j["proj4_string"] = proj4_str;
  j["longitude"]    = origin_lonlat.x();
  j["latitude"]     = origin_lonlat.y();

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    DLOG(FATAL) << "Failed to open file: " << filename;
    return;
  }
  ofs << j.dump(4) << std::endl;
}

void LoadLAS(const std::string &filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, Eigen::Vector2d &offset, std::string &proj_str) {
  cloud.reset(new pcl::PointCloud<pcl::PointXYZI>);

  pdal::StageFactory factory;
  pdal::Stage *reader = factory.createStage("readers.las");
  pdal::Options opts;
  opts.add(pdal::Option("filename", filename));
  reader->setOptions(opts);

  pdal::PointTable table;
  reader->prepare(table);
  pdal::PointViewSet viewSet = reader->execute(table);
  pdal::PointViewPtr view    = *viewSet.begin();

  pdal::MetadataNode metadata = reader->getMetadata();
  offset.x()                  = metadata.findChild("offset_x").value<double>();
  offset.y()                  = metadata.findChild("offset_y").value<double>();
  proj_str                    = metadata.findChild("srs").findChild("proj4").value<std::string>();

  DLOG(INFO) << std::fixed << std::setprecision(9) << "LAS offset: X=" << offset.x() << ", Y=" << offset.y();

  cloud->width    = view->size();
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width);

  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZI p;
    p.x              = view->getFieldAs<double>(pdal::Dimension::Id::X, i) - offset.x();
    p.y              = view->getFieldAs<double>(pdal::Dimension::Id::Y, i) - offset.y();
    p.z              = view->getFieldAs<double>(pdal::Dimension::Id::Z, i);
    p.intensity      = view->getFieldAs<float>(pdal::Dimension::Id::Intensity, i);
    cloud->points[i] = p;
  }
}

void SavePointCloud(const std::string &filename, const pcl::PointCloud<pcl::PointXYZI> &cloud, const std::vector<Eigen::Vector3f> &normals) {
  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud.size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud.points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    cloud_out->points.push_back(np);
  }
}

void PcaEstimateNormalNoDirect(const pcl::PointCloud<pcl::PointXYZI>::ConstPtr &cloud, int k, double downsample_voxel_size,
                               std::vector<Eigen::Vector3f> &normals) {
  normals.resize(cloud->size());

  DLOG(INFO) << "Building kdtree for normal estimation...";
  pcl::KdTreeFLANN<pcl::PointXYZI>::Ptr tree(new pcl::KdTreeFLANN<pcl::PointXYZI>);
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
  DownsamplePointCloudInternal<pcl::PointXYZI>(*cloud, *cloud_downsampled, downsample_voxel_size);
  tree->setInputCloud(cloud_downsampled);

  DLOG(INFO) << "Estimating normals...";
#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(k);
    std::vector<float> k_sqr_distances(k);

    if (tree->nearestKSearch(cloud->at(i), k, k_indices, k_sqr_distances) <= 0) {
      continue;
    }

    Eigen::Vector3d centroid = Eigen::Vector3d::Zero();
    for (int j : k_indices) {
      centroid += cloud_downsampled->points[j].getVector3fMap().cast<double>();
    }
    centroid /= k_indices.size();

    Eigen::Matrix3d covariance = Eigen::Matrix3d::Zero();
    for (int j : k_indices) {
      Eigen::Vector3d neighbor = cloud_downsampled->points[j].getVector3fMap().cast<double>();
      Eigen::Vector3d cp       = neighbor - centroid;
      covariance += cp * cp.transpose();
    }

    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3d> eigensolver(covariance);
    Eigen::Matrix3d eigenvectors = eigensolver.eigenvectors();

    normals[i] = eigenvectors.col(0).cast<float>();
  }
}

void SmoothPointCloud(const std::vector<Eigen::Vector3f> &normals, int kNearestNeighbors, double max_search_radius, double sigma_d, double sigma_n,
                      pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud) {
  pcl::search::KdTree<pcl::PointXYZI>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZI>);
  tree->setInputCloud(cloud);

#pragma omp parallel for
  for (int i = 0; i < cloud->size(); ++i) {
    std::vector<int> k_indices(kNearestNeighbors);
    std::vector<float> k_sqr_distances(kNearestNeighbors);

    if (tree->radiusSearch(cloud->at(i), max_search_radius, k_indices, k_sqr_distances, kNearestNeighbors) <= 0) {
      continue;
    }

    double delta_p = 0;
    double sum_w   = 0;
    auto p         = cloud->points[i].getVector3fMap();
    for (int j : k_indices) {
      auto q     = cloud->points[j].getVector3fMap();
      double d_d = (q - p).norm();
      double d_n = (q - p).dot(normals[i]);
      double w   = std::exp(-d_d * d_d / (2 * sigma_d * sigma_d) - d_n * d_n / (2 * sigma_n * sigma_n));
      delta_p += w * d_n;
      sum_w += w;
    }
    p += delta_p / sum_w * normals[i];
  }
}

void SaveToLocalENU(const std::string &filename, const pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud_nooffset, const LocalENUTransformer &transformer) {
  DLOG(INFO) << "Transforming to local ENU coordinates, origin (lat, lon): " << transformer.getOriginLonLat().x() << ", "
             << transformer.getOriginLonLat().y();
  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_out(new pcl::PointCloud<pcl::PointXYZI>);
  cloud_out->resize(cloud_nooffset->size());
  cloud_out->width    = cloud_nooffset->width;
  cloud_out->height   = cloud_nooffset->height;
  cloud_out->is_dense = cloud_nooffset->is_dense;
  for (int i = 0; i < cloud_nooffset->size(); ++i) {
    Eigen::Vector2d localEN        = transformer.Transform(cloud_nooffset->points[i].getVector2fMap().cast<double>() + transformer.getOffset());
    cloud_out->points[i].x         = localEN.x();
    cloud_out->points[i].y         = localEN.y();
    cloud_out->points[i].z         = cloud_nooffset->points[i].z;
    cloud_out->points[i].intensity = cloud_nooffset->points[i].intensity;
  }

  DLOG(INFO) << "Saving to " << filename;
  pcl::io::savePCDFileBinary(filename, *cloud_out);
}

void SaveToLocalENU(const std::string &filename, const std::string &initial_pose_filename, const Eigen::Vector2d &offset,
                    const LocalENUTransformer &transformer) {
  std::ifstream file(initial_pose_filename);
  CHECK(file) << initial_pose_filename;

  std::ofstream outfile(filename);
  CHECK(outfile) << filename;

  std::string line;
  std::getline(file, line);
  outfile << line << std::endl;

  auto delta_rot = Eigen::AngleAxisd(-transformer.GetMeridianConvergence() / 180.0 * M_PI, Eigen::Vector3d(0, 0, 1));

  std::string file_path;
  double tx, ty, tz, roll, pitch, yaw, qx, qy, qz, qw, timestamp;
  while (file >> file_path >> tx >> ty >> tz >> roll >> pitch >> yaw >> qx >> qy >> qz >> qw >> timestamp) {
    Eigen::Vector3d pos(tx - offset.x(), ty - offset.y(), tz);
    Eigen::Quaterniond rot(qw, qx, qy, qz);

    Eigen::Vector2d transformed = transformer.Transform(Eigen::Vector2d(tx, ty));

    // Eigen::Vector3d pos_enu    = delta_rot * pos;
    Eigen::Quaterniond rot_enu = delta_rot * rot;

    outfile << std::fixed << std::setprecision(9) << file_path << " " << transformed.x() << " " << transformed.y() << " " << tz << " " << 0 << " "
            << 0 << " " << 0 << " " << rot_enu.x() << " " << rot_enu.y() << " " << rot_enu.z() << " " << rot_enu.w() << " " << timestamp << std::endl;
  }
}

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = true;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 6, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_nooffset(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<Eigen::Vector3f> normals;

  DCHECK(boost::filesystem::is_regular_file(FLAGS_las_filename));

  DLOG(INFO) << "Loading LAS file...";
  Eigen::Vector2d offset;
  std::string proj_str;
  LoadLAS(FLAGS_las_filename, cloud_nooffset, offset, proj_str);
  DLOG(INFO) << "Loaded " << cloud_nooffset->size() << " points.";

  if (!FLAGS_output_full) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    DownsamplePointCloudInternal<pcl::PointXYZI>(*cloud_nooffset, *cloud_downsampled, kDownsampleVoxelSize);
    cloud_nooffset = cloud_downsampled;
    DLOG(INFO) << "Downsampled to " << cloud_nooffset->size() << " points.";
  }

  LocalENUTransformer enu_transformer(offset, proj_str);
  DLOG(INFO) << "Meridian convergence at origin: " << enu_transformer.GetMeridianConvergence() << " degrees.";

  DLOG(INFO) << "Save to " << FLAGS_las_filename + "_offset.json";
  SaveLasOffsetJson(FLAGS_las_filename + "_offset.json", offset, proj_str, enu_transformer.getOriginLonLat());

  SaveToLocalENU(FLAGS_las_filename + "_normals_localenu.pcd", cloud_nooffset, enu_transformer);
  SaveToLocalENU(FLAGS_initial_pose_filename + "_localenu.txt", FLAGS_initial_pose_filename, offset, enu_transformer);

  std::cout << "done." << std::endl;

  return 0;
}
