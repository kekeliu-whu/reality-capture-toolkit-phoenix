#include "navvis_recon/slam_batch_collator.hpp"

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

namespace navvis_recon::slam {
namespace {

constexpr std::int64_t kMinimumDurationNs = 50'000'000;
constexpr std::uint64_t kMinimumRawRayCount = 58'000;
constexpr std::int64_t kConservativeScanSupportNs = 60'000'000;

struct PacketEvent {
  std::int64_t timestamp_ns = 0;
  std::uint32_t ray_count = 0;
};

struct TimedPoint {
  std::int64_t timestamp_ns = 0;
  Eigen::Vector3f point = Eigen::Vector3f::Zero();
  Eigen::Vector3f origin = Eigen::Vector3f::Zero();
};

template<typename T>
T loadScalar(const std::byte* bytes) noexcept {
  T value{};
  std::memcpy(&value, bytes, sizeof(value));
  return value;
}

std::vector<PacketEvent> mergePacketEvents(
    const std::array<std::vector<PacketEvent>, 2>& by_sensor) {
  std::vector<PacketEvent> output;
  output.reserve(by_sensor[0].size() + by_sensor[1].size());
  std::merge(
      by_sensor[0].begin(), by_sensor[0].end(), by_sensor[1].begin(),
      by_sensor[1].end(), std::back_inserter(output),
      [](const PacketEvent& left, const PacketEvent& right) {
        return left.timestamp_ns < right.timestamp_ns;
      });
  return output;
}

std::vector<std::int64_t> accumulateBoundaries(
    const std::vector<PacketEvent>& packets,
    const std::int64_t first_all_sources_ns) {
  const auto first = std::lower_bound(
      packets.begin(), packets.end(), first_all_sources_ns,
      [](const PacketEvent& packet, const std::int64_t timestamp_ns) {
        return packet.timestamp_ns < timestamp_ns;
      });
  if (first == packets.end()) {
    throw std::runtime_error("no collated packets remain after lidar startup");
  }
  std::vector<std::int64_t> output;
  output.reserve(static_cast<std::size_t>(
      (packets.back().timestamp_ns - first->timestamp_ns) /
          kMinimumDurationNs +
      1));
  std::int64_t batch_start_ns = first->timestamp_ns;
  std::int64_t previous_timestamp_ns = batch_start_ns;
  std::uint64_t accumulated_ray_count = 0;
  for (auto packet = first; packet != packets.end(); ++packet) {
    if (packet->timestamp_ns - batch_start_ns >= kMinimumDurationNs &&
        accumulated_ray_count >= kMinimumRawRayCount) {
      output.push_back(previous_timestamp_ns);
      batch_start_ns = packet->timestamp_ns;
      accumulated_ray_count = 0;
    }
    accumulated_ray_count += packet->ray_count;
    previous_timestamp_ns = packet->timestamp_ns;
  }
  return output;
}

}  // namespace

class SlamBatchCollator::Implementation {
 public:
  explicit Implementation(const std::filesystem::path& path) : archive(path) {
    if (archive.version() != SlamArchiveVersion::V6) {
      throw std::invalid_argument(
          "autonomous SLAM batching requires an NVSLAM6 archive");
    }
    const auto& records = archive.records();
    if (records.empty()) {
      throw std::invalid_argument("SLAM scan archive is empty");
    }
    record_timestamps.reserve(records.size());
    std::array<std::vector<PacketEvent>, 2> packets_by_sensor;
    std::array<bool, 2> has_first{false, false};
    std::array<std::int64_t, 2> first_record_ns{};
    for (std::size_t index = 0; index < records.size(); ++index) {
      const SlamScanRecord& record = records[index];
      record_timestamps.push_back(record.timestamp_ns);
      if (!has_first[record.sensor]) {
        has_first[record.sensor] = true;
        first_record_ns[record.sensor] =
            (record.timestamp_ns / 1000) * 1000;
      }
      if (record.packet_count == 0U) {
        continue;
      }
      if (record.ray_count % record.packet_count != 0U) {
        throw std::runtime_error(
            "SLAM raw ray slots are not uniform within a scan");
      }
      const std::uint32_t rays_per_packet =
          record.ray_count / record.packet_count;
      const SlamScanView scan = archive.scan(index);
      auto& output = packets_by_sensor[record.sensor];
      for (std::size_t packet = 0; packet < record.packet_count; ++packet) {
        output.push_back(PacketEvent{
            scan.packet_timestamps_ns.at(packet), rays_per_packet});
      }
    }
    if (!has_first[0] || !has_first[1]) {
      throw std::runtime_error("dual-laser SLAM archive is missing a sensor");
    }
    for (const auto& packets : packets_by_sensor) {
      if (!std::is_sorted(
              packets.begin(), packets.end(),
              [](const PacketEvent& left, const PacketEvent& right) {
                return left.timestamp_ns < right.timestamp_ns;
              })) {
        throw std::runtime_error(
            "per-sensor SLAM packet timestamps are not ordered");
      }
    }
    first_all_sources_ns = std::max(first_record_ns[0], first_record_ns[1]);
    const std::vector<PacketEvent> packets =
        mergePacketEvents(packets_by_sensor);
    batch_timestamps = accumulateBoundaries(packets, first_all_sources_ns);
    if (batch_timestamps.empty()) {
      throw std::runtime_error("SLAM packet accumulator produced no batches");
    }
  }

  SlamScanArchive archive;
  std::vector<std::int64_t> record_timestamps;
  std::vector<std::int64_t> batch_timestamps;
  std::int64_t first_all_sources_ns = 0;
  std::size_t next_batch = 0;

  std::vector<TimedPoint> collectSensor(
      const std::uint8_t sensor, const std::int64_t window_start_ns,
      const std::int64_t batch_end_ns, const bool first_batch) const {
    const auto& records = archive.records();
    const auto first_record = std::lower_bound(
        record_timestamps.begin(), record_timestamps.end(),
        window_start_ns - kConservativeScanSupportNs);
    std::vector<TimedPoint> output;
    for (std::size_t record_index =
             static_cast<std::size_t>(first_record - record_timestamps.begin());
         record_index < records.size(); ++record_index) {
      const SlamScanRecord& record = records[record_index];
      if (record.timestamp_ns >
          batch_end_ns + archive.recordHeaderSupportNs()) {
        break;
      }
      if (record.sensor != sensor ||
          record.timestamp_ns + kConservativeScanSupportNs < window_start_ns) {
        continue;
      }
      const SlamScanView scan = archive.scan(record_index);
      const Eigen::Vector3f origin(scan.ray_origins.at(0U, 0U),
                                   scan.ray_origins.at(0U, 1U),
                                   scan.ray_origins.at(0U, 2U));
      const std::byte* const point_bytes = scan.points.rawData();
      const std::byte* const timestamp_bytes =
          scan.point_timestamps_ns.rawData();
      const std::size_t point_stride = scan.points.rowStrideBytes();
      const std::size_t timestamp_stride =
          scan.point_timestamps_ns.strideBytes();
      const std::int64_t tolerance_ns =
          archive.timestampBoundaryToleranceNs();
      for (std::size_t point_index = 0; point_index < record.point_count;
           ++point_index) {
        const std::int64_t timestamp_ns = loadScalar<std::int64_t>(
            timestamp_bytes + point_index * timestamp_stride);
        const bool keep = first_batch
                              ? timestamp_ns >=
                                        window_start_ns - tolerance_ns &&
                                    timestamp_ns <= batch_end_ns + tolerance_ns
                              : timestamp_ns >
                                        window_start_ns + tolerance_ns &&
                                    timestamp_ns <= batch_end_ns + tolerance_ns;
        if (!keep) {
          continue;
        }
        const std::byte* const values =
            point_bytes + point_index * point_stride;
        output.push_back(TimedPoint{
            std::min(timestamp_ns, batch_end_ns),
            Eigen::Vector3f(loadScalar<float>(values),
                            loadScalar<float>(values + sizeof(float)),
                            loadScalar<float>(values + 2U * sizeof(float))),
            origin});
      }
    }
    if (!std::is_sorted(
            output.begin(), output.end(),
            [](const TimedPoint& left, const TimedPoint& right) {
              return left.timestamp_ns < right.timestamp_ns;
            })) {
      std::stable_sort(
          output.begin(), output.end(),
          [](const TimedPoint& left, const TimedPoint& right) {
            return left.timestamp_ns < right.timestamp_ns;
          });
    }
    return output;
  }

  TimedRangeBatch next() {
    if (next_batch >= batch_timestamps.size()) {
      throw std::out_of_range("SLAM batch collator is exhausted");
    }
    const std::int64_t end_ns = batch_timestamps[next_batch];
    const std::int64_t window_start_ns =
        next_batch == 0U
            ? first_all_sources_ns
            : std::max(batch_timestamps[next_batch - 1U],
                       end_ns - kMinimumDurationNs);
    std::vector<TimedPoint> horizontal =
        collectSensor(0U, window_start_ns, end_ns, next_batch == 0U);
    std::vector<TimedPoint> vertical =
        collectSensor(1U, window_start_ns, end_ns, next_batch == 0U);
    if (horizontal.empty() && vertical.empty()) {
      throw std::runtime_error("no raw laser rays were assigned to SLAM batch " +
                               std::to_string(next_batch));
    }

    std::vector<TimedPoint> merged;
    merged.reserve(horizontal.size() + vertical.size());
    std::merge(
        horizontal.begin(), horizontal.end(), vertical.begin(), vertical.end(),
        std::back_inserter(merged),
        [](const TimedPoint& left, const TimedPoint& right) {
          return left.timestamp_ns < right.timestamp_ns;
        });
    TimedRangeBatch output;
    output.timestamp_ns = end_ns;
    output.points.reserve(merged.size());
    output.origins.reserve(merged.size());
    output.point_timestamps_ns.reserve(merged.size());
    for (const TimedPoint& point : merged) {
      output.points.push_back(point.point);
      output.origins.push_back(point.origin);
      output.point_timestamps_ns.push_back(point.timestamp_ns);
    }
    ++next_batch;
    return output;
  }
};

SlamBatchCollator::SlamBatchCollator(const std::filesystem::path& archive_path)
    : implementation_(std::make_unique<Implementation>(archive_path)) {}

SlamBatchCollator::~SlamBatchCollator() = default;
SlamBatchCollator::SlamBatchCollator(SlamBatchCollator&&) noexcept = default;
SlamBatchCollator& SlamBatchCollator::operator=(SlamBatchCollator&&) noexcept =
    default;

std::size_t SlamBatchCollator::batchCount() const noexcept {
  return implementation_->batch_timestamps.size();
}

std::size_t SlamBatchCollator::nextBatchIndex() const noexcept {
  return implementation_->next_batch;
}

const std::vector<std::int64_t>& SlamBatchCollator::batchTimestampsNs() const
    noexcept {
  return implementation_->batch_timestamps;
}

std::int64_t SlamBatchCollator::firstAllSourcesTimestampNs() const noexcept {
  return implementation_->first_all_sources_ns;
}

bool SlamBatchCollator::hasNext() const noexcept {
  return implementation_->next_batch < implementation_->batch_timestamps.size();
}

TimedRangeBatch SlamBatchCollator::next() { return implementation_->next(); }

}  // namespace navvis_recon::slam
