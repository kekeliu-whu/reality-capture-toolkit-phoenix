#include <array>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>

float fromBits(std::uint32_t bits) {
    float value;
    std::memcpy(&value, &bits, sizeof(value));
    return value;
}

std::uint32_t bits(float value) {
    std::uint32_t result;
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::array<float, 4> accumulate(const std::array<std::array<int, 3>, 5>& rgb) {
    const std::array<std::uint32_t, 5> weight_bits{
        0x358637bd, 0x431deb63, 0x44333472, 0x4481ffe0, 0x446fa2bb};
    std::array<float, 4> result{};
    for (std::size_t rank = 0; rank < rgb.size(); ++rank) {
        const float weight = fromBits(weight_bits[rank]);
        result[0] += static_cast<float>(rgb[rank][0]) * weight;
        result[1] += static_cast<float>(rgb[rank][1]) * weight;
        result[2] += static_cast<float>(rgb[rank][2]) * weight;
        result[3] += weight;
    }
    return result;
}

int main() {
    // Semantic PLY order: red, green, blue.
    const std::array<std::array<int, 3>, 5> frozen{{
        {71, 57, 36}, {10, 19, 14}, {127, 104, 36}, {34, 62, 26}, {30, 52, 19}}};
    auto current = frozen;
    current[2][0] = 128;
    current[2][2] = 37;
    for (const auto& [label, values] :
         std::array<std::pair<const char*, std::array<float, 4>>, 2>{{
             {"frozen", accumulate(frozen)}, {"current", accumulate(current)}}}) {
        std::cout << label;
        for (float value : values) {
            std::cout << " 0x" << std::hex << bits(value) << std::dec << '/' << std::setprecision(17)
                      << value;
        }
        const float inverse = 1.0F / values[3];
        std::cout << " inv=0x" << std::hex << bits(inverse) << std::dec;
        for (std::size_t channel = 0; channel < 3; ++channel) {
            const float normalized = values[channel] * inverse;
            std::cout << " out" << channel << "=0x" << std::hex << bits(normalized) << std::dec
                      << '/' << std::setprecision(17) << normalized << '/' << std::lrintf(normalized);
        }
        std::cout << '\n';
    }
}
