// Offline clean-room probe for the installed RegionPitchRange predicate.
//
// Build this as a shared object and preload it only while running the frozen
// vendor acceptance executable.  The weak, exported virtual function is
// interposable, so the probe can record the exact float point/object operands
// and then delegate the decision to the original implementation.  It is not a
// production dependency of the reconstruction pipeline.

#include <dlfcn.h>
#include <fcntl.h>
#include <unistd.h>

#include <array>
#include <atomic>
#include <cerrno>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace {

constexpr char kSymbol[] =
    "_ZNK6navvis8geometry16RegionPitchRangeIfE8containsERKN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEE";

using ContainsFunction = bool (*)(const void*, const float*);

struct CaptureRecord {
    std::uint32_t magic;
    std::uint32_t sequence;
    float center;
    float half_width;
    std::array<float, 3> point;
    std::uint8_t result;
    std::uint8_t reserved[3];
};

static_assert(sizeof(CaptureRecord) == 32);

std::atomic<std::uint32_t> sequence{0};

float environmentFloat(const char* name, float fallback) {
    const char* text = std::getenv(name);
    if (text == nullptr || *text == '\0') {
        return fallback;
    }
    char* end = nullptr;
    errno = 0;
    const float value = std::strtof(text, &end);
    return errno == 0 && end != text ? value : fallback;
}

ContainsFunction originalContains() {
    static const auto function = reinterpret_cast<ContainsFunction>(
        dlsym(RTLD_NEXT, kSymbol));
    return function;
}

bool matchesTarget(const float* point) {
    const float target_x = environmentFloat("NAVVIS_PITCH_TARGET_X", -0.48254255F);
    const float target_y = environmentFloat("NAVVIS_PITCH_TARGET_Y", 1.10080658F);
    const float target_z = environmentFloat("NAVVIS_PITCH_TARGET_Z", -0.27190988F);
    const float tolerance = environmentFloat("NAVVIS_PITCH_TARGET_TOLERANCE", 0.002F);
    return std::abs(point[0] - target_x) <= tolerance &&
           std::abs(point[1] - target_y) <= tolerance &&
           std::abs(point[2] - target_z) <= tolerance;
}

void appendRecord(const void* owner, const float* point, bool result) {
    const char* output = std::getenv("NAVVIS_PITCH_CAPTURE_OUTPUT");
    if (output == nullptr || *output == '\0') {
        return;
    }
    const auto* object = static_cast<const std::uint8_t*>(owner);
    CaptureRecord record{};
    record.magic = 0x48435450U;  // "PTCH" in little endian.
    record.sequence = sequence.fetch_add(1, std::memory_order_relaxed);
    std::memcpy(&record.center, object + 8, sizeof(float));
    std::memcpy(&record.half_width, object + 12, sizeof(float));
    std::memcpy(record.point.data(), point, 3 * sizeof(float));
    record.result = result ? 1U : 0U;

    const int descriptor = open(output, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0644);
    if (descriptor >= 0) {
        const ssize_t ignored = write(descriptor, &record, sizeof(record));
        (void)ignored;
        close(descriptor);
    }
}

}  // namespace

extern "C" bool navvisRegionPitchContains(const void* owner, const float* point)
    asm("_ZNK6navvis8geometry16RegionPitchRangeIfE8containsERKN5Eigen6MatrixIfLi3ELi1ELi0ELi3ELi1EEE");

extern "C" bool navvisRegionPitchContains(const void* owner, const float* point) {
    const ContainsFunction original = originalContains();
    if (original == nullptr) {
        _exit(127);
    }
    const bool result = original(owner, point);
    if (matchesTarget(point)) {
        appendRecord(owner, point, result);
        if (environmentFloat("NAVVIS_PITCH_EXIT_AFTER_CAPTURE", 1.0F) != 0.0F) {
            _exit(86);
        }
    }
    return result;
}
