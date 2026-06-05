#pragma once

#include <cstdint>
#include <cstdlib>
#include <limits>
#include <stdexcept>
#include <string>

namespace sfm_phoenix {
namespace internal {

inline bool EnvVarEnabled(const char* name) {
#ifdef _WIN32
    char* value = nullptr;
    size_t value_size = 0;
    const errno_t error = _dupenv_s(&value, &value_size, name);
    const bool enabled = (error == 0) && value != nullptr && value[0] != '\0';
    std::free(value);
    return enabled;
#else
    const char* value = std::getenv(name);
    return value != nullptr && value[0] != '\0';
#endif
}

inline int CheckedIntCast(const int64_t value, const char* context) {
    if (value < std::numeric_limits<int>::min() ||
        value > std::numeric_limits<int>::max()) {
        throw std::runtime_error(std::string(context) + " exceeds int range");
    }
    return static_cast<int>(value);
}

}  // namespace internal
}  // namespace sfm_phoenix