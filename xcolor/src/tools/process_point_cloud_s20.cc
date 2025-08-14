#include <glog/logging.h>
#include <omp.h>
#include <pcl/filters/bilateral.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/search/kdtree.h>
#include <Eigen/Eigen>
#include <boost/filesystem.hpp>
#include <fstream>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/StageFactory.hpp>
#include <thread>
#include <vector>
#include <nlohmann/json.hpp>

DEFINE_string(las_filename, "D:/ProjectX/project-3d/data/sfm/area3/sfm/colorized.las", "Input project path");
DEFINE_bool(output_full, false, "Output full point cloud");

static constexpr double kDownsampleVoxelSize    = 0.05;
static constexpr int kNearestNeighbors          = 15;
static constexpr int kSmoothMaxNearestNeighbors = 100;
static constexpr double kSmoothMaxSearchRadius  = 0.3;
static constexpr double kSmoothSigmaD           = 0.05;
static constexpr double kSmoothSigmaN           = 0.05;

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

void SaveLasOffsetJson(const std::string &filename, double offset_x, double offset_y, const std::string &proj4_str) {
  nlohmann::json j;
  j["offset_x"] = offset_x;
  j["offset_y"] = offset_y;
  j["proj4_string"] = proj4_str;

  std::ofstream ofs(filename);
  if (!ofs.is_open()) {
    DLOG(FATAL) << "Failed to open file: " << filename;
    return;
  }
  ofs << j.dump(4) << std::endl;
}

void LoadLAS(const std::string &filename, pcl::PointCloud<pcl::PointXYZI>::Ptr &cloud, double &offset_x, double &offset_y, std::string &proj_str) {
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
  offset_x                    = metadata.findChild("offset_x").value<double>();
  offset_y                    = metadata.findChild("offset_y").value<double>();
  proj_str                    = metadata.findChild("srs").findChild("proj4").value<std::string>();

  DLOG(INFO) << std::fixed << std::setprecision(9) << "LAS offset: X=" << offset_x << ", Y=" << offset_y;

  cloud->width    = view->size();
  cloud->height   = 1;
  cloud->is_dense = false;
  cloud->points.resize(cloud->width);

  for (size_t i = 0; i < view->size(); ++i) {
    pcl::PointXYZI p;
    p.x              = view->getFieldAs<double>(pdal::Dimension::Id::X, i) - offset_x;
    p.y              = view->getFieldAs<double>(pdal::Dimension::Id::Y, i) - offset_y;
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

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = true;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  pcl::PointCloud<pcl::PointXYZI>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZI>);
  std::vector<Eigen::Vector3f> normals;

  DCHECK(boost::filesystem::is_regular_file(FLAGS_las_filename));

  DLOG(INFO) << "Loading LAS file...";
  double offset_x, offset_y;
  std::string proj_str;
  LoadLAS(FLAGS_las_filename, cloud, offset_x, offset_y, proj_str);
  DLOG(INFO) << "Loaded " << cloud->size() << " points.";

  if (!FLAGS_output_full) {
    pcl::PointCloud<pcl::PointXYZI>::Ptr cloud_downsampled(new pcl::PointCloud<pcl::PointXYZI>);
    DownsamplePointCloudInternal<pcl::PointXYZI>(*cloud, *cloud_downsampled, kDownsampleVoxelSize);
    cloud = cloud_downsampled;
    DLOG(INFO) << "Downsampled to " << cloud->size() << " points.";
  }

  DLOG(INFO) << "Computing normals...";
  PcaEstimateNormalNoDirect(cloud, kNearestNeighbors, kDownsampleVoxelSize, normals);

  DLOG(INFO) << "Smoothing...";
  // pcl::io::savePCDFileBinary(FLAGS_project_output_path + "/before-smooth.pcd", *cloud);
  SmoothPointCloud(normals, kSmoothMaxNearestNeighbors, kSmoothMaxSearchRadius, kSmoothSigmaD, kSmoothSigmaN, cloud);

  pcl::PointCloud<pcl::PointXYZINormal>::Ptr cloud_with_normals(new pcl::PointCloud<pcl::PointXYZINormal>);
  for (int i = 0; i < cloud->size(); ++i) {
    pcl::PointXYZINormal np;
    np.getVector3fMap()       = cloud->points[i].getVector3fMap();
    np.getNormalVector3fMap() = normals[i];
    np.intensity              = cloud->points[i].intensity;
    cloud_with_normals->push_back(np);
  }

  DLOG(INFO) << "Saving cloud with normals...";
  pcl::io::savePCDFileBinary(FLAGS_las_filename + "_normals.pcd", *cloud_with_normals);
  DLOG(INFO) << "Save to " << FLAGS_las_filename + "_normals.pcd";

  DLOG(INFO) << "Save to " << FLAGS_las_filename + "_offset.json";
  SaveLasOffsetJson(FLAGS_las_filename + "_offset.json", offset_x, offset_y, proj_str);

  std::cout << "done." << std::endl;

  return 0;
}
