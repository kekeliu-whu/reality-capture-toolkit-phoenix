
#include <colmap/scene/reconstruction.h>
#include <glog/logging.h>
#include <omp.h>
#include <pcl/io/ply_io.h>

#include "common/types.h"
#include "core/utils.h"
#include "core/xcolor_lib.h"
#include "io/colmap_io.h"

DEFINE_string(images_path, "D:/BaiduNetdiskDownload/s10-colmap/images", "Images path");
DEFINE_string(sfm_result_path, "D:/BaiduNetdiskDownload/s10-colmap", "SFM databaset filename");
DEFINE_string(point_cloud_filename, "D:/BaiduNetdiskDownload/s10-colmap/uncolorized.ply", "Point cloud filename");
DEFINE_string(output_path, "D:/BaiduNetdiskDownload/s10-colmap", "Output path");

int main(int argc, char **argv) {
  gflags::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging(argv[0]);

  FLAGS_logtostderr = 1;

  int cores      = std::thread::hardware_concurrency();
  int cores_used = std::max(cores - 4, 1);
  DLOG(INFO) << "Using " << cores_used << "/" << cores << " cores.";
  omp_set_dynamic(0);
  omp_set_num_threads(cores_used);

  std::vector<xcolor::Image> images;

  // loading images and camera parameters
  xcolor::PrintMemoryUsage();
  xcolor::ReadImages(FLAGS_sfm_result_path, FLAGS_images_path, images);
  xcolor::PrintMemoryUsage();

  // loading point cloud
  DLOG(INFO) << "Loading point cloud from " << FLAGS_point_cloud_filename << " ...";
  pcl::PointCloud<pcl::PointXYZRGB> cloud;
  int ret = pcl::io::loadPLYFile(FLAGS_point_cloud_filename, cloud);
  xcolor::PrintMemoryUsage();
  CHECK_EQ(ret, 0) << "Load point cloud failed: " << FLAGS_point_cloud_filename;
  DLOG(INFO) << "Load " << cloud.size() << " points.";

  DLOG(INFO) << "Start xcolor...";
  xcolor::PerformXColor(cloud, images, FLAGS_output_path);
  DLOG(INFO) << "Finish xcolor.";

  return 0;
}
