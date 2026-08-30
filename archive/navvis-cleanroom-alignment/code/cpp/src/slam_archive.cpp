#include "navvis_recon/slam_archive.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstring>
#ifdef _WIN32
#define NOMINMAX
#include <Windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#endif
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>

namespace navvis_recon {
namespace {

constexpr std::size_t kMagicSize = 8U;
constexpr std::size_t kHeaderV1V2Size = 13U;  // <BdI
constexpr std::size_t kHeaderV3V4Size = 17U;  // <BdII
constexpr std::size_t kHeaderV5Size = 17U;    // <BqII
constexpr std::size_t kHeaderV6Size = 21U;    // <BqIII
constexpr std::size_t kPointNsSize = 20U;     // <fffq

constexpr std::array<std::array<char, kMagicSize>, 6> kMagic{{
    {{'N', 'V', 'S', 'L', 'A', 'M', '1', '\0'}},
    {{'N', 'V', 'S', 'L', 'A', 'M', '2', '\0'}},
    {{'N', 'V', 'S', 'L', 'A', 'M', '3', '\0'}},
    {{'N', 'V', 'S', 'L', 'A', 'M', '4', '\0'}},
    {{'N', 'V', 'S', 'L', 'A', 'M', '5', '\0'}},
    {{'N', 'V', 'S', 'L', 'A', 'M', '6', '\0'}},
}};

#pragma pack(push, 1)
struct HeaderV1V2Layout {
    std::uint8_t sensor;
    double timestamp_seconds;
    std::uint32_t point_count;
};

struct HeaderV3V4Layout {
    std::uint8_t sensor;
    double timestamp_seconds;
    std::uint32_t ray_count;
    std::uint32_t point_count;
};

struct HeaderV5Layout {
    std::uint8_t sensor;
    std::int64_t timestamp_ns;
    std::uint32_t ray_count;
    std::uint32_t point_count;
};

struct HeaderV6Layout {
    std::uint8_t sensor;
    std::int64_t timestamp_ns;
    std::uint32_t ray_count;
    std::uint32_t packet_count;
    std::uint32_t point_count;
};

struct PointNsLayout {
    float x;
    float y;
    float z;
    std::int64_t timestamp_ns;
};
#pragma pack(pop)

static_assert(sizeof(HeaderV1V2Layout) == kHeaderV1V2Size);
static_assert(sizeof(HeaderV3V4Layout) == kHeaderV3V4Size);
static_assert(sizeof(HeaderV5Layout) == kHeaderV5Size);
static_assert(sizeof(HeaderV6Layout) == kHeaderV6Size);
static_assert(sizeof(PointNsLayout) == kPointNsSize);
static_assert(sizeof(float) == 4U && std::numeric_limits<float>::is_iec559);
static_assert(sizeof(double) == 8U && std::numeric_limits<double>::is_iec559);
static_assert(sizeof(std::int64_t) == 8U);

bool hostIsLittleEndian() noexcept {
    const std::uint16_t value = 1U;
    return *reinterpret_cast<const std::uint8_t*>(&value) == 1U;
}

template<typename T>
T loadScalar(const std::byte* data) noexcept {
    T value{};
    std::memcpy(&value, data, sizeof(value));
    return value;
}

std::int64_t roundTiesToEven(double value, const char* context) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(std::string(context) + " is not finite");
    }
    const double lower = std::floor(value);
    const double fraction = value - lower;
    double rounded = lower;
    if (fraction > 0.5 || (fraction == 0.5 && std::fmod(std::abs(lower), 2.0) == 1.0)) {
        rounded = lower + 1.0;
    }
    if (rounded < static_cast<double>(std::numeric_limits<std::int64_t>::min()) ||
        rounded >= -static_cast<double>(std::numeric_limits<std::int64_t>::min())) {
        throw std::runtime_error(std::string(context) + " is outside int64 range");
    }
    return static_cast<std::int64_t>(rounded);
}

std::uint64_t checkedAdvance(
    std::uint64_t offset, std::uint64_t count, std::uint64_t item_size,
    std::uint64_t file_size, const char* context) {
    if (item_size != 0U && count > (std::numeric_limits<std::uint64_t>::max() - offset) /
                                      item_size) {
        throw std::runtime_error(std::string(context) + " byte range overflows uint64");
    }
    const std::uint64_t end = offset + count * item_size;
    if (end > file_size) {
        throw std::runtime_error(std::string("truncated ") + context);
    }
    return end;
}

std::runtime_error systemFailure(const std::string& operation) {
#ifdef _WIN32
    const DWORD code = GetLastError();
    char* buffer = nullptr;
    const DWORD length = FormatMessageA(
        FORMAT_MESSAGE_ALLOCATE_BUFFER | FORMAT_MESSAGE_FROM_SYSTEM |
            FORMAT_MESSAGE_IGNORE_INSERTS,
        nullptr, code, 0, reinterpret_cast<char*>(&buffer), 0, nullptr);
    std::string message = length != 0U && buffer != nullptr
                              ? std::string(buffer, length)
                              : "Windows error " + std::to_string(code);
    if (buffer != nullptr) {
        LocalFree(buffer);
    }
    return std::runtime_error(operation + ": " + message);
#else
    return std::runtime_error(operation + ": " + std::strerror(errno));
#endif
}

const std::array<float, 6>& sensorOrigins() {
    // Exact float32 bit patterns used by surveyor_frontend.py. In particular,
    // horizontal x is the provider's transform round-trip value 0xa4f20000.
    static const std::array<float, 6> values = [] {
        constexpr std::array<std::uint32_t, 6> bits{{
            0xa4f20000U, 0x3d9ba5e3U, 0x3cc985f0U,
            0xba6d9327U, 0x3ee5f652U, 0xbf22146dU,
        }};
        std::array<float, 6> result{};
        for (std::size_t index = 0; index < bits.size(); ++index) {
            std::memcpy(&result[index], &bits[index], sizeof(float));
        }
        return result;
    }();
    return values;
}

struct FormatDescription {
    std::size_t header_size = 0U;
    std::size_t point_stride = 0U;
    bool has_normals = false;
    PerRayTimestampView::Encoding timestamp_encoding =
        PerRayTimestampView::Encoding::None;
    std::size_t timestamp_offset = 0U;
};

FormatDescription describeFormat(SlamArchiveVersion version) {
    switch (version) {
        case SlamArchiveVersion::V1:
            return {kHeaderV1V2Size, 6U * sizeof(float), true,
                    PerRayTimestampView::Encoding::None, 0U};
        case SlamArchiveVersion::V2:
            return {kHeaderV1V2Size, 7U * sizeof(float), true,
                    PerRayTimestampView::Encoding::Float32SecondsFromScan,
                    6U * sizeof(float)};
        case SlamArchiveVersion::V3:
            return {kHeaderV3V4Size, 7U * sizeof(float), true,
                    PerRayTimestampView::Encoding::Float32SecondsFromScan,
                    6U * sizeof(float)};
        case SlamArchiveVersion::V4:
            return {kHeaderV3V4Size, 4U * sizeof(float), false,
                    PerRayTimestampView::Encoding::Float32SecondsFromScan,
                    3U * sizeof(float)};
        case SlamArchiveVersion::V5:
            return {kHeaderV5Size, kPointNsSize, false,
                    PerRayTimestampView::Encoding::Int64Nanoseconds,
                    3U * sizeof(float)};
        case SlamArchiveVersion::V6:
            return {kHeaderV6Size, kPointNsSize, false,
                    PerRayTimestampView::Encoding::Int64Nanoseconds,
                    3U * sizeof(float)};
    }
    throw std::runtime_error("unsupported NVSLAM archive version");
}

}  // namespace

struct SlamScanArchive::Impl {
    ~Impl() {
#ifdef _WIN32
        if (mapping != nullptr) {
            UnmapViewOfFile(mapping);
        }
        if (mapping_handle != nullptr) {
            CloseHandle(mapping_handle);
        }
        if (file_handle != INVALID_HANDLE_VALUE) {
            CloseHandle(file_handle);
        }
#else
        if (mapping != MAP_FAILED) {
            ::munmap(mapping, mapping_size);
        }
        if (file_descriptor >= 0) {
            ::close(file_descriptor);
        }
#endif
    }

    std::filesystem::path path;
#ifdef _WIN32
    HANDLE file_handle = INVALID_HANDLE_VALUE;
    HANDLE mapping_handle = nullptr;
    void* mapping = nullptr;
#else
    int file_descriptor = -1;
    void* mapping = MAP_FAILED;
#endif
    std::size_t mapping_size = 0U;
    std::uint64_t file_size = 0U;
    SlamArchiveVersion version = SlamArchiveVersion::V1;
    FormatDescription format;
    std::vector<SlamScanRecord> records;

    [[nodiscard]] const std::byte* bytes() const noexcept {
        return static_cast<const std::byte*>(mapping);
    }
};

std::int64_t PerRayTimestampView::at(std::size_t index) const {
    if (index >= size_) {
        throw std::out_of_range("SLAM archive timestamp index is out of range");
    }
    const std::byte* value = data_ + index * stride_;
    switch (encoding_) {
        case Encoding::Int64Nanoseconds:
            return loadScalar<std::int64_t>(value);
        case Encoding::Float32SecondsFromScan: {
            const float offset_seconds = loadScalar<float>(value);
            const std::int64_t offset_ns = roundTiesToEven(
                static_cast<double>(offset_seconds) * 1.0e9,
                "SLAM per-ray timestamp offset");
            if ((offset_ns > 0 &&
                 scan_timestamp_ns_ > std::numeric_limits<std::int64_t>::max() - offset_ns) ||
                (offset_ns < 0 &&
                 scan_timestamp_ns_ < std::numeric_limits<std::int64_t>::min() - offset_ns)) {
                throw std::runtime_error("SLAM per-ray timestamp overflows int64");
            }
            return scan_timestamp_ns_ + offset_ns;
        }
        case Encoding::None:
            break;
    }
    throw std::logic_error("SLAM archive has no per-ray timestamps");
}

std::vector<std::int64_t> PerRayTimestampView::copyNanoseconds() const {
    std::vector<std::int64_t> result(size_);
    for (std::size_t index = 0; index < size_; ++index) {
        result[index] = at(index);
    }
    return result;
}

SlamScanArchive::SlamScanArchive(const std::filesystem::path& path)
    : impl_(std::make_unique<Impl>()) {
    if (!hostIsLittleEndian()) {
        throw std::runtime_error("NVSLAM archives require a little-endian host");
    }

    impl_->path = path;
#ifdef _WIN32
    impl_->file_handle = CreateFileW(
        path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING,
        FILE_ATTRIBUTE_NORMAL, nullptr);
    if (impl_->file_handle == INVALID_HANDLE_VALUE) {
        throw systemFailure("cannot open SLAM scan archive " + path.string());
    }
    if (GetFileType(impl_->file_handle) != FILE_TYPE_DISK) {
        throw std::runtime_error("SLAM scan archive is not a regular file: " + path.string());
    }
    LARGE_INTEGER status_size{};
    if (!GetFileSizeEx(impl_->file_handle, &status_size)) {
        throw systemFailure("cannot stat SLAM scan archive " + path.string());
    }
    if (status_size.QuadPart < 0 ||
        static_cast<std::uint64_t>(status_size.QuadPart) >
            std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("SLAM scan archive size is unsupported");
    }
    impl_->file_size = static_cast<std::uint64_t>(status_size.QuadPart);
#else
    impl_->file_descriptor = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
    if (impl_->file_descriptor < 0) {
        throw systemFailure("cannot open SLAM scan archive " + path.string());
    }

    struct stat status {};
    if (::fstat(impl_->file_descriptor, &status) != 0) {
        throw systemFailure("cannot stat SLAM scan archive " + path.string());
    }
    if (!S_ISREG(status.st_mode)) {
        throw std::runtime_error("SLAM scan archive is not a regular file: " + path.string());
    }
    if (status.st_size < 0 || static_cast<std::uint64_t>(status.st_size) >
                                  std::numeric_limits<std::size_t>::max()) {
        throw std::runtime_error("SLAM scan archive size is unsupported");
    }
    impl_->file_size = static_cast<std::uint64_t>(status.st_size);
#endif
    impl_->mapping_size = static_cast<std::size_t>(impl_->file_size);
    if (impl_->mapping_size < kMagicSize) {
        throw std::runtime_error("truncated SLAM scan archive magic");
    }

#ifdef _WIN32
    impl_->mapping_handle = CreateFileMappingW(
        impl_->file_handle, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (impl_->mapping_handle == nullptr) {
        throw systemFailure("cannot create SLAM scan archive mapping " + path.string());
    }
    impl_->mapping = MapViewOfFile(
        impl_->mapping_handle, FILE_MAP_READ, 0, 0, impl_->mapping_size);
    if (impl_->mapping == nullptr) {
        throw systemFailure("cannot map SLAM scan archive " + path.string());
    }
#else
    impl_->mapping = ::mmap(
        nullptr, impl_->mapping_size, PROT_READ, MAP_PRIVATE, impl_->file_descriptor, 0);
    if (impl_->mapping == MAP_FAILED) {
        throw systemFailure("cannot mmap SLAM scan archive " + path.string());
    }
#endif

    const std::byte* bytes = impl_->bytes();
    bool recognized = false;
    for (std::size_t index = 0; index < kMagic.size(); ++index) {
        if (std::memcmp(bytes, kMagic[index].data(), kMagicSize) == 0) {
            impl_->version = static_cast<SlamArchiveVersion>(index + 1U);
            recognized = true;
            break;
        }
    }
    if (!recognized) {
        throw std::runtime_error("not an NVSLAM1--NVSLAM6 scan archive");
    }
    impl_->format = describeFormat(impl_->version);

    std::uint64_t offset = kMagicSize;
    std::size_t source_index = 0U;
    while (offset < impl_->file_size) {
        const std::uint64_t source_file_offset = offset;
        offset = checkedAdvance(
            offset, 1U, impl_->format.header_size, impl_->file_size,
            "SLAM scan record header");
        const std::byte* header = bytes + source_file_offset;

        SlamScanRecord record;
        record.sensor = loadScalar<std::uint8_t>(header);
        record.source_file_offset = source_file_offset;
        record.source_index = source_index++;
        if (record.sensor > 1U) {
            throw std::runtime_error(
                "unsupported SLAM laser index " + std::to_string(record.sensor));
        }

        if (impl_->version == SlamArchiveVersion::V1 ||
            impl_->version == SlamArchiveVersion::V2) {
            const double timestamp_seconds = loadScalar<double>(header + 1U);
            record.timestamp_ns = roundTiesToEven(
                timestamp_seconds * 1.0e9, "SLAM record timestamp");
            record.point_count = loadScalar<std::uint32_t>(header + 9U);
            record.ray_count = record.point_count;
        } else if (impl_->version == SlamArchiveVersion::V3 ||
                   impl_->version == SlamArchiveVersion::V4) {
            const double timestamp_seconds = loadScalar<double>(header + 1U);
            record.timestamp_ns = roundTiesToEven(
                timestamp_seconds * 1.0e9, "SLAM record timestamp");
            record.ray_count = loadScalar<std::uint32_t>(header + 9U);
            record.point_count = loadScalar<std::uint32_t>(header + 13U);
        } else {
            record.timestamp_ns = loadScalar<std::int64_t>(header + 1U);
            record.ray_count = loadScalar<std::uint32_t>(header + 9U);
            if (impl_->version == SlamArchiveVersion::V5) {
                record.point_count = loadScalar<std::uint32_t>(header + 13U);
            } else {
                record.packet_count = loadScalar<std::uint32_t>(header + 13U);
                record.point_count = loadScalar<std::uint32_t>(header + 17U);
            }
        }
        if (record.point_count > record.ray_count) {
            throw std::runtime_error("SLAM scan point count exceeds raw ray count");
        }
        if (record.packet_count > record.ray_count) {
            throw std::runtime_error("SLAM scan packet count exceeds raw ray count");
        }

        record.packet_timestamps_offset = offset;
        offset = checkedAdvance(
            offset, record.packet_count, sizeof(std::int64_t), impl_->file_size,
            "SLAM packet timestamp array");
        record.data_offset = offset;
        offset = checkedAdvance(
            offset, record.point_count, impl_->format.point_stride, impl_->file_size,
            "SLAM scan point array");
        impl_->records.push_back(record);
    }

    std::stable_sort(
        impl_->records.begin(), impl_->records.end(),
        [](const SlamScanRecord& left, const SlamScanRecord& right) {
            return left.timestamp_ns < right.timestamp_ns;
        });

#ifdef _WIN32
    LARGE_INTEGER final_size{};
    if (!GetFileSizeEx(impl_->file_handle, &final_size)) {
        throw systemFailure("cannot re-stat SLAM scan archive " + path.string());
    }
    if (final_size.QuadPart != status_size.QuadPart) {
        throw std::runtime_error("SLAM scan archive changed while metadata was read");
    }
#else
    struct stat final_status {};
    if (::fstat(impl_->file_descriptor, &final_status) != 0) {
        throw systemFailure("cannot re-stat SLAM scan archive " + path.string());
    }
    if (final_status.st_size != status.st_size) {
        throw std::runtime_error("SLAM scan archive changed while metadata was read");
    }
#endif
}

SlamScanArchive::~SlamScanArchive() = default;
SlamScanArchive::SlamScanArchive(SlamScanArchive&&) noexcept = default;
SlamScanArchive& SlamScanArchive::operator=(SlamScanArchive&&) noexcept = default;

const std::filesystem::path& SlamScanArchive::path() const noexcept { return impl_->path; }

SlamArchiveVersion SlamScanArchive::version() const noexcept { return impl_->version; }

std::uint64_t SlamScanArchive::fileSize() const noexcept { return impl_->file_size; }

const std::vector<SlamScanRecord>& SlamScanArchive::records() const noexcept {
    return impl_->records;
}

std::size_t SlamScanArchive::recordCount() const noexcept { return impl_->records.size(); }

bool SlamScanArchive::hasNormals() const noexcept { return impl_->format.has_normals; }

bool SlamScanArchive::hasPointTimestamps() const noexcept {
    return impl_->format.timestamp_encoding != PerRayTimestampView::Encoding::None;
}

bool SlamScanArchive::hasStoredPacketTimestamps() const noexcept {
    return impl_->version == SlamArchiveVersion::V6;
}

std::int64_t SlamScanArchive::timestampBoundaryToleranceNs() const noexcept {
    return impl_->version == SlamArchiveVersion::V5 || impl_->version == SlamArchiveVersion::V6
               ? 1
               : 0;
}

std::int64_t SlamScanArchive::recordHeaderSupportNs() const noexcept {
    return impl_->version == SlamArchiveVersion::V5 || impl_->version == SlamArchiveVersion::V6
               ? 1000
               : 0;
}

SlamScanView SlamScanArchive::scan(std::size_t ordered_index) const {
    if (ordered_index >= impl_->records.size()) {
        throw std::out_of_range("SLAM scan record index is out of range");
    }
    const SlamScanRecord& record = impl_->records[ordered_index];
    const std::byte* data = impl_->bytes() + record.data_offset;

    SlamScanView view;
    view.points = StridedArrayView<float>(
        data, record.point_count, 3U, impl_->format.point_stride, sizeof(float));
    if (impl_->format.has_normals) {
        view.normals = StridedArrayView<float>(
            data + 3U * sizeof(float), record.point_count, 3U,
            impl_->format.point_stride, sizeof(float));
    }

    const auto& origins = sensorOrigins();
    view.ray_origins = StridedArrayView<float>(
        reinterpret_cast<const std::byte*>(origins.data() + 3U * record.sensor),
        record.point_count, 3U, 0U, sizeof(float));

    if (impl_->format.timestamp_encoding != PerRayTimestampView::Encoding::None) {
        view.point_timestamps_ns = PerRayTimestampView(
            impl_->format.timestamp_encoding, data + impl_->format.timestamp_offset,
            record.point_count, impl_->format.point_stride, record.timestamp_ns);
    }
    view.packet_timestamps_ns = StridedArrayView<std::int64_t>(
        impl_->bytes() + record.packet_timestamps_offset, record.packet_count, 1U,
        sizeof(std::int64_t), sizeof(std::int64_t));
    return view;
}

}  // namespace navvis_recon
