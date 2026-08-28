#pragma once

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <type_traits>
#include <vector>

namespace navvis_recon {

enum class SlamArchiveVersion : std::uint8_t {
    V1 = 1,
    V2 = 2,
    V3 = 3,
    V4 = 4,
    V5 = 5,
    V6 = 6,
};

// A non-owning view over a row-major, possibly strided array. The view points
// either into the archive's read-only mapping or into immutable static data.
// It remains valid while the owning mapping (including a moved-to archive)
// is alive.
template<typename T>
class StridedArrayView {
    static_assert(std::is_trivially_copyable<T>::value, "array values must be POD-like");

public:
    StridedArrayView() = default;

    [[nodiscard]] const std::byte* rawData() const noexcept { return data_; }
    [[nodiscard]] std::size_t rows() const noexcept { return rows_; }
    [[nodiscard]] std::size_t columns() const noexcept { return columns_; }
    [[nodiscard]] std::size_t rowStrideBytes() const noexcept { return row_stride_; }
    [[nodiscard]] std::size_t columnStrideBytes() const noexcept { return column_stride_; }
    [[nodiscard]] bool empty() const noexcept { return rows_ == 0U || columns_ == 0U; }

    [[nodiscard]] bool isContiguous() const noexcept {
        return column_stride_ == sizeof(T) &&
               (rows_ <= 1U || row_stride_ == columns_ * sizeof(T));
    }

    [[nodiscard]] T at(std::size_t row, std::size_t column = 0U) const {
        if (row >= rows_ || column >= columns_) {
            throw std::out_of_range("SLAM archive array index is out of range");
        }
        T value{};
        std::memcpy(
            &value, data_ + row * row_stride_ + column * column_stride_, sizeof(value));
        return value;
    }

    // Materialization is intentionally bounded to this view (normally one
    // scan), never the complete archive.
    [[nodiscard]] std::vector<T> copy() const {
        if (columns_ != 0U && rows_ > std::numeric_limits<std::size_t>::max() / columns_) {
            throw std::length_error("SLAM archive array is too large to copy");
        }
        std::vector<T> result(rows_ * columns_);
        for (std::size_t row = 0; row < rows_; ++row) {
            for (std::size_t column = 0; column < columns_; ++column) {
                result[row * columns_ + column] = at(row, column);
            }
        }
        return result;
    }

private:
    friend class SlamScanArchive;

    StridedArrayView(
        const std::byte* data, std::size_t rows, std::size_t columns,
        std::size_t row_stride, std::size_t column_stride) noexcept
        : data_(data),
          rows_(rows),
          columns_(columns),
          row_stride_(row_stride),
          column_stride_(column_stride) {}

    const std::byte* data_ = nullptr;
    std::size_t rows_ = 0U;
    std::size_t columns_ = 0U;
    std::size_t row_stride_ = 0U;
    std::size_t column_stride_ = 0U;
};

class PerRayTimestampView {
public:
    enum class Encoding : std::uint8_t {
        None,
        Float32SecondsFromScan,
        Int64Nanoseconds,
    };

    PerRayTimestampView() = default;

    [[nodiscard]] Encoding encoding() const noexcept { return encoding_; }
    [[nodiscard]] const std::byte* rawData() const noexcept { return data_; }
    [[nodiscard]] std::size_t size() const noexcept { return size_; }
    [[nodiscard]] std::size_t strideBytes() const noexcept { return stride_; }
    [[nodiscard]] bool empty() const noexcept { return size_ == 0U; }
    [[nodiscard]] std::int64_t at(std::size_t index) const;
    [[nodiscard]] std::vector<std::int64_t> copyNanoseconds() const;

private:
    friend class SlamScanArchive;

    PerRayTimestampView(
        Encoding encoding, const std::byte* data, std::size_t size, std::size_t stride,
        std::int64_t scan_timestamp_ns) noexcept
        : encoding_(encoding),
          data_(data),
          size_(size),
          stride_(stride),
          scan_timestamp_ns_(scan_timestamp_ns) {}

    Encoding encoding_ = Encoding::None;
    const std::byte* data_ = nullptr;
    std::size_t size_ = 0U;
    std::size_t stride_ = 0U;
    std::int64_t scan_timestamp_ns_ = 0;
};

struct SlamScanRecord {
    std::uint8_t sensor = 0U;
    std::int64_t timestamp_ns = 0;
    std::uint32_t ray_count = 0U;
    std::uint32_t packet_count = 0U;
    std::uint32_t point_count = 0U;
    std::uint64_t packet_timestamps_offset = 0U;
    std::uint64_t data_offset = 0U;
    std::uint64_t source_file_offset = 0U;
    std::size_t source_index = 0U;
};

struct SlamScanView {
    StridedArrayView<float> points;
    std::optional<StridedArrayView<float>> normals;
    StridedArrayView<float> ray_origins;
    PerRayTimestampView point_timestamps_ns;
    StridedArrayView<std::int64_t> packet_timestamps_ns;
};

// Read-only reader for the compact NVSLAM1--NVSLAM6 scan archives emitted by
// the clean-room Pandar path. On-disk integers and IEEE-754 values are little
// endian and packed without ABI padding. Unsupported hosts and malformed
// archives are rejected instead of being interpreted heuristically.
class SlamScanArchive {
public:
    explicit SlamScanArchive(const std::filesystem::path& path);
    ~SlamScanArchive();

    SlamScanArchive(const SlamScanArchive&) = delete;
    SlamScanArchive& operator=(const SlamScanArchive&) = delete;
    SlamScanArchive(SlamScanArchive&&) noexcept;
    SlamScanArchive& operator=(SlamScanArchive&&) noexcept;

    [[nodiscard]] const std::filesystem::path& path() const noexcept;
    [[nodiscard]] SlamArchiveVersion version() const noexcept;
    [[nodiscard]] std::uint64_t fileSize() const noexcept;
    [[nodiscard]] const std::vector<SlamScanRecord>& records() const noexcept;
    [[nodiscard]] std::size_t recordCount() const noexcept;

    [[nodiscard]] bool hasNormals() const noexcept;
    [[nodiscard]] bool hasPointTimestamps() const noexcept;
    [[nodiscard]] bool hasStoredPacketTimestamps() const noexcept;
    [[nodiscard]] std::int64_t timestampBoundaryToleranceNs() const noexcept;
    [[nodiscard]] std::int64_t recordHeaderSupportNs() const noexcept;

    // Records are indexed in stable nondecreasing timestamp order, matching
    // surveyor_frontend.py::SlamScanArchive. source_index retains file order.
    [[nodiscard]] SlamScanView scan(std::size_t ordered_index) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace navvis_recon
