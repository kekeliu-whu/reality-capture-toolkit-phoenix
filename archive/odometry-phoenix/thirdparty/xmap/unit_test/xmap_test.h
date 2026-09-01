#include <utility>

#include "xmap.h"

namespace xmap {
class XmapTest {
 public:
  std::unordered_map<VoxelLoc, SmallVoxelValue>& getSmallVoxelMap(Xmap& xmap) {
    return xmap.small_voxel_map_;
  }

  void setConfigs(Xmap& xmap, Configs configs) { xmap.configs_ = std::move(configs); }
};
}  // namespace xmap
