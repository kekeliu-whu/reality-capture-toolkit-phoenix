//
// Created by youyuan on 24-1-3.
//

#include "voxel_loc.h"

namespace xmap {

VoxelLoc::VoxelLoc() : x_(0), y_(0), z_(0) {}

VoxelLoc::VoxelLoc(IntDataType vx, IntDataType vy, IntDataType vz) : x_(vx), y_(vy), z_(vz) {}

VoxelLoc::VoxelLoc(const V3F& point, FloatDataType voxel_size) {
  Eigen::Matrix<FloatDataType, 3, 1> loc_xyz_d = point / voxel_size;
  Eigen::Matrix<IntDataType, 3, 1> loc_xyz_i;
  loc_xyz_i << static_cast<IntDataType>(std::round(loc_xyz_d(0))),
      static_cast<IntDataType>(std::round(loc_xyz_d(1))),
      static_cast<IntDataType>(std::round(loc_xyz_d(2)));
  x_ = loc_xyz_i.x();
  y_ = loc_xyz_i.y();
  z_ = loc_xyz_i.z();
}

bool VoxelLoc::operator==(const VoxelLoc& other) const {
  return (x_ == other.x_ && y_ == other.y_ && z_ == other.z_);
}

bool VoxelLoc::operator!=(const VoxelLoc& other) const {
  return (x_ != other.x_ || y_ != other.y_ || z_ != other.z_);
}

std::ostream& operator<<(std::ostream& os, const VoxelLoc& obj) {
  os << obj.x_ << "_" << obj.y_ << "_" << obj.z_;
  return os;
}

VoxelLoc VoxelLoc::operator-(const VoxelLoc& other) const {
  return {x_ - other.x_, y_ - other.y_, z_ - other.z_};
}

std::string VoxelLoc::toString() const {
  std::ostringstream oss;
  oss << x_ << "_" << y_ << "_" << z_;
  return oss.str();
}
}  // namespace xmap
