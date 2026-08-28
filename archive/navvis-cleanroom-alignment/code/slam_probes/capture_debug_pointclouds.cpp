// Read-only interposer for the installed SLAM debug point-cloud callback.
//
// The matcher already constructs named source/target/correspondence clouds.
// Capturing that boundary avoids inferring its data layout from visual output.

#include <Eigen/Core>

#include <atomic>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fcntl.h>
#include <optional>
#include <string>
#include <unistd.h>
#include <vector>

namespace navvis::core {
struct UniversalTimeScaleClock {
  using rep = std::int64_t;
  using period = std::nano;
  using duration = std::chrono::nanoseconds;
  using time_point = std::chrono::time_point<UniversalTimeScaleClock>;
  static constexpr bool is_steady = false;
};
}  // namespace navvis::core

namespace navvis::pointcloud {

class DebugVisualizationInterface {
 public:
  void visualizePointCloud(
      const std::string& group, const std::string& name,
      const std::vector<Eigen::Vector3f>& points,
      std::optional<navvis::core::UniversalTimeScaleClock::time_point> stamp);
};

void DebugVisualizationInterface::visualizePointCloud(
    const std::string& group, const std::string& name,
    const std::vector<Eigen::Vector3f>& points,
    std::optional<navvis::core::UniversalTimeScaleClock::time_point> stamp) {
  using Function = void (*)(
      DebugVisualizationInterface*, const std::string&, const std::string&,
      const std::vector<Eigen::Vector3f>&,
      std::optional<navvis::core::UniversalTimeScaleClock::time_point>);
  static const auto original = reinterpret_cast<Function>(dlsym(
      RTLD_NEXT,
      "_ZN6navvis10pointcloud27DebugVisualizationInterface19visualizePointCloud"
      "ERKNSt7__cxx1112basic_stringIcSt11char_traitsIcESaIcEEES9_"
      "RKSt6vectorIN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEESaISD_EE"
      "St8optionalINSt6chrono10time_pointINS_4core23UniversalTimeScaleClockE"
      "NSJ_8durationIlSt5ratioILl1ELl1000000000EEEEEEE"));

  const char* directory = std::getenv("NAVVIS_DEBUG_CLOUD_CAPTURE_DIR");
  if (directory != nullptr && directory[0] != '\0') {
    static std::atomic<unsigned long> sequence{0};
    const unsigned long index = sequence.fetch_add(1);
    std::string safe = group + "__" + name;
    for (char& value : safe) {
      if (!((value >= 'a' && value <= 'z') ||
            (value >= 'A' && value <= 'Z') ||
            (value >= '0' && value <= '9') || value == '_' || value == '-')) {
        value = '_';
      }
    }
    char path[4096];
    std::snprintf(path, sizeof(path), "%s/%06lu_%s.bin", directory, index,
                  safe.c_str());
    const int fd = open(path, O_WRONLY | O_CREAT | O_TRUNC, 0600);
    if (fd >= 0) {
      struct Header {
        char magic[8];
        std::int64_t timestamp_ns;
        std::uint64_t point_count;
        std::uint32_t group_size;
        std::uint32_t name_size;
      } header{{'N', 'V', 'D', 'B', 'G', 'P', 'C', '1'},
               stamp ? stamp->time_since_epoch().count() : 0,
               points.size(),
               static_cast<std::uint32_t>(group.size()),
               static_cast<std::uint32_t>(name.size())};
      (void)write(fd, &header, sizeof(header));
      (void)write(fd, group.data(), group.size());
      (void)write(fd, name.data(), name.size());
      if (!points.empty()) {
        (void)write(fd, points.data(), points.size() * sizeof(Eigen::Vector3f));
      }
      close(fd);
    }
  }
  if (original != nullptr) {
    original(this, group, name, points, stamp);
  }
}

}  // namespace navvis::pointcloud
