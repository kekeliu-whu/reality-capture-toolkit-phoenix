#pragma once

#include "navvis_recon/slam_archive.hpp"
#include "navvis_recon/slam_frontend.hpp"

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <memory>
#include <vector>

namespace navvis_recon::slam {

// Reconstructs the dual-Pandar 50 ms / 58,000 raw-slot batches directly from
// NVSLAM6 packet metadata. Rays are emitted in stable time order with sensor 0
// before sensor 1 on equal timestamps, matching RangeDataCollator.
class SlamBatchCollator {
 public:
  explicit SlamBatchCollator(const std::filesystem::path& archive_path);
  ~SlamBatchCollator();
  SlamBatchCollator(SlamBatchCollator&&) noexcept;
  SlamBatchCollator& operator=(SlamBatchCollator&&) noexcept;
  SlamBatchCollator(const SlamBatchCollator&) = delete;
  SlamBatchCollator& operator=(const SlamBatchCollator&) = delete;

  [[nodiscard]] std::size_t batchCount() const noexcept;
  [[nodiscard]] std::size_t nextBatchIndex() const noexcept;
  [[nodiscard]] const std::vector<std::int64_t>& batchTimestampsNs() const noexcept;
  [[nodiscard]] std::int64_t firstAllSourcesTimestampNs() const noexcept;
  [[nodiscard]] bool hasNext() const noexcept;
  TimedRangeBatch next();

 private:
  class Implementation;
  std::unique_ptr<Implementation> implementation_;
};

}  // namespace navvis_recon::slam
