#include "navvis_recon/slam_archive.hpp"

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

std::size_t parseExpectedCount(const char* text) {
    std::size_t consumed = 0U;
    const unsigned long long value = std::stoull(text, &consumed);
    if (text[consumed] != '\0' || value > std::numeric_limits<std::size_t>::max()) {
        throw std::invalid_argument("invalid expected record count");
    }
    return static_cast<std::size_t>(value);
}

void require(bool condition, const std::string& message) {
    if (!condition) {
        throw std::runtime_error(message);
    }
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 3) {
            std::cerr << "usage: slam_archive_test ARCHIVE [EXPECTED_RECORD_COUNT]\n";
            return EXIT_FAILURE;
        }

        const navvis_recon::SlamScanArchive archive(argv[1]);
        const auto& records = archive.records();
        require(!records.empty(), "archive contains no scan records");
        if (argc == 3) {
            require(
                records.size() == parseExpectedCount(argv[2]),
                "record count does not match EXPECTED_RECORD_COUNT");
        }

        std::vector<bool> source_indices(records.size(), false);
        std::vector<const navvis_recon::SlamScanRecord*> source_records(records.size());
        std::uint64_t total_rays = 0U;
        std::uint64_t total_packets = 0U;
        std::uint64_t total_points = 0U;
        std::size_t sampled_values = 0U;

        for (std::size_t index = 0; index < records.size(); ++index) {
            const navvis_recon::SlamScanRecord& record = records[index];
            require(record.sensor <= 1U, "record has unsupported sensor index");
            require(record.point_count <= record.ray_count, "record has more points than rays");
            require(record.source_index < records.size(), "record source index is out of range");
            require(!source_indices[record.source_index], "record source index is duplicated");
            source_indices[record.source_index] = true;
            source_records[record.source_index] = &record;
            if (index != 0U) {
                require(
                    records[index - 1U].timestamp_ns <= record.timestamp_ns,
                    "records are not in nondecreasing timestamp order");
                if (records[index - 1U].timestamp_ns == record.timestamp_ns) {
                    require(
                        records[index - 1U].source_index < record.source_index,
                        "equal-timestamp records do not preserve file order");
                }
            }

            const navvis_recon::SlamScanView scan = archive.scan(index);
            require(
                scan.points.rows() == record.point_count && scan.points.columns() == 3U,
                "point view shape does not match record metadata");
            require(
                scan.ray_origins.rows() == record.point_count &&
                    scan.ray_origins.columns() == 3U &&
                    scan.ray_origins.rowStrideBytes() == 0U,
                "ray-origin broadcast view is invalid");
            require(
                scan.normals.has_value() == archive.hasNormals(),
                "normal availability does not match archive version");
            if (scan.normals) {
                require(
                    scan.normals->rows() == record.point_count &&
                        scan.normals->columns() == 3U,
                    "normal view shape does not match record metadata");
            }
            if (archive.hasPointTimestamps()) {
                require(
                    scan.point_timestamps_ns.size() == record.point_count,
                    "per-ray timestamp count does not match record metadata");
            } else {
                require(
                    scan.point_timestamps_ns.empty(),
                    "archive version unexpectedly exposes per-ray timestamps");
            }
            require(
                scan.packet_timestamps_ns.rows() == record.packet_count &&
                    scan.packet_timestamps_ns.columns() == 1U,
                "packet timestamp view shape does not match record metadata");

            if (record.point_count != 0U) {
                const std::size_t last = record.point_count - 1U;
                for (const std::size_t point_index : {std::size_t{0U}, last}) {
                    for (std::size_t axis = 0; axis < 3U; ++axis) {
                        require(
                            std::isfinite(scan.points.at(point_index, axis)),
                            "sampled point is not finite");
                        require(
                            std::isfinite(scan.ray_origins.at(point_index, axis)),
                            "sampled ray origin is not finite");
                        require(
                            scan.ray_origins.at(point_index, axis) ==
                                scan.ray_origins.at(0U, axis),
                            "ray-origin broadcast changed within a scan");
                        if (scan.normals) {
                            require(
                                std::isfinite(scan.normals->at(point_index, axis)),
                                "sampled normal is not finite");
                        }
                        ++sampled_values;
                    }
                    if (archive.hasPointTimestamps()) {
                        static_cast<void>(scan.point_timestamps_ns.at(point_index));
                    }
                }
            }
            if (record.packet_count != 0U) {
                static_cast<void>(scan.packet_timestamps_ns.at(0U));
                static_cast<void>(scan.packet_timestamps_ns.at(record.packet_count - 1U));
            }

            total_rays += record.ray_count;
            total_packets += record.packet_count;
            total_points += record.point_count;
        }
        for (const bool present : source_indices) {
            require(present, "record source indices are not a complete permutation");
        }
        require(
            source_records.front()->source_file_offset == 8U,
            "first source record does not immediately follow the archive magic");
        for (std::size_t index = 1U; index < source_records.size(); ++index) {
            require(
                source_records[index - 1U]->source_file_offset <
                    source_records[index]->source_file_offset,
                "source record offsets are not strictly increasing");
        }

        // Exercise bounded materialization on one scan. This must not copy or
        // touch the complete multi-gigabyte archive.
        const navvis_recon::SlamScanView first_scan = archive.scan(0U);
        const std::vector<float> copied_points = first_scan.points.copy();
        require(
            copied_points.size() == records.front().point_count * 3ULL,
            "bounded point copy has the wrong size");
        if (archive.hasPointTimestamps()) {
            const std::vector<std::int64_t> copied_timestamps =
                first_scan.point_timestamps_ns.copyNanoseconds();
            require(
                copied_timestamps.size() == records.front().point_count,
                "bounded timestamp copy has the wrong size");
        }

        std::cout << "PASS"
                  << " version=" << static_cast<unsigned>(archive.version())
                  << " bytes=" << archive.fileSize()
                  << " records=" << records.size()
                  << " rays=" << total_rays
                  << " packets=" << total_packets
                  << " points=" << total_points
                  << " sampled_scalars=" << sampled_values
                  << " first_timestamp_ns=" << records.front().timestamp_ns
                  << " last_timestamp_ns=" << records.back().timestamp_ns << '\n';
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
