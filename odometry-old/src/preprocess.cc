#include "preprocess.h"

#define RETURN0 0x00
#define RETURN0AND1 0x10

Preprocess::Preprocess() : blind(0.01) {
  SCAN_RATE         = 10;
  given_offset_time = false;
}

Preprocess::~Preprocess() {}

void Preprocess::process(const std::shared_ptr<proto::LidarMsg> &msg, PointCloudXYZI::Ptr &pcl_out) {
  avia_handler(msg);
  *pcl_out = pl_surf;
}

void Preprocess::avia_handler(const std::shared_ptr<proto::LidarMsg> &msg) {
  pl_surf.clear();
  pl_full.clear();
  double t1     = omp_get_wtime();
  int    plsize = msg->points().size();

  pl_surf.reserve(plsize);
  pl_full.resize(plsize);

  for (uint32_t i = 1; i < plsize; i++) {
    auto &pt = msg->points().at(i);
    pl_full[i].x         = pt.x();
    pl_full[i].y         = pt.y();
    pl_full[i].z         = pt.z();
    pl_full[i].intensity = pt.intensity();
    pl_full[i].curvature = (pt.timestamp() - msg->points().at(0).timestamp())* 1000; // use curvature as time of each laser points, curvature unit: ms

    if ((abs(pl_full[i].x - pl_full[i - 1].x) > 1e-7) || (abs(pl_full[i].y - pl_full[i - 1].y) > 1e-7) ||
        (abs(pl_full[i].z - pl_full[i - 1].z) > 1e-7) &&
            (pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z >
              (blind * blind))) {
      double dist = sqrt(pl_full[i].x * pl_full[i].x + pl_full[i].y * pl_full[i].y + pl_full[i].z * pl_full[i].z);
      if (pt.x() <= 0 && dist < 1.5 && abs(pl_full[i].y) < 1.5) {
        continue;
      }
      // todo kk fov

      pl_surf.push_back(pl_full[i]);
    }
  }
}
