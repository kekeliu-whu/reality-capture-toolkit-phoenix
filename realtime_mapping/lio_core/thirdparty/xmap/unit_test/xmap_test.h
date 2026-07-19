#include <utility>

#include "xmap.h"

namespace xmap {
class XmapTest {
 public:
  void dynamicSaveTest(Xmap& xmap, const V3F& view_point) { xmap.dynamicSave(view_point); }

  void dynamicLoadTest(Xmap& xmap, const V3F& view_point) { xmap.dynamicLoad(view_point); }

  std::unordered_map<VoxelLoc, SmallVoxelValue>& getSmallVoxelMap(Xmap& xmap) {
    return xmap.small_voxel_map_;
  }

  std::unordered_map<VoxelLoc, std::unordered_set<VoxelLoc>>& getLargeVoxelMap(Xmap& xmap) {
    return xmap.large_voxel_map_;
  }

  void setConfigs(Xmap& xmap, Configs configs) { xmap.configs_ = std::move(configs); }
};
}  // namespace xmap