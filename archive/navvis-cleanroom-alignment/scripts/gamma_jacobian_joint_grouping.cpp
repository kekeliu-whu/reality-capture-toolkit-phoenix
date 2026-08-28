#include <ceres/jet.h>

#include <array>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <string_view>

namespace {

using Jet = ceres::Jet<double, 4>;

struct Evaluation {
    std::array<Jet, 2> residuals{};
};

double normalizedByte(std::uint8_t value) {
    constexpr float inverse_255 = 1.0F / 255.0F;
    return static_cast<double>(static_cast<float>(value) * inverse_255);
}

float normalizedQuality(std::uint16_t value) {
    return static_cast<float>(value) / 65535.0F;
}

Evaluation evaluate(bool square_first) {
    constexpr std::array<std::uint8_t, 2> intensities{130U, 153U};
    constexpr std::array<std::uint16_t, 2> packed_qualities{57199U, 7499U};

    std::array<Jet, 2> gains{Jet(1.0), Jet(1.0)};
    std::array<Jet, 2> exponents{Jet(1.0), Jet(1.0)};
    gains[0].v[0] = 1.0;
    exponents[0].v[1] = 1.0;
    gains[1].v[2] = 1.0;
    exponents[1].v[3] = 1.0;

    std::array<Jet, 2> corrected{};
    Jet total_weight(0.0);
    Jet weighted_intensity(0.0);
    for (std::size_t index = 0; index < 2; ++index) {
        using std::pow;
        const double input = normalizedByte(intensities[index]);
        corrected[index] = gains[index] * pow(Jet(input), exponents[index]);
        const Jet weight(normalizedQuality(packed_qualities[index]));
        total_weight += weight;
        weighted_intensity += weight * corrected[index];
    }
    const Jet mean = weighted_intensity / total_weight;

    Evaluation result;
    for (std::size_t row = 0; row < 2; ++row) {
        const Jet difference = corrected[row] - mean;
        const Jet weight(normalizedQuality(packed_qualities[row]));
        result.residuals[row] = square_first ? weight * (difference * difference)
                                             : weight * difference * difference;
    }
    return result;
}

void printScalar(std::string_view grouping, std::string_view field, int row, int column,
                 double value) {
    std::uint64_t bits = 0;
    static_assert(sizeof(bits) == sizeof(value));
    std::memcpy(&bits, &value, sizeof(bits));
    std::cout << grouping << ' ' << field << ' ' << row << ' ' << column << ' '
              << std::hexfloat << value << " 0x" << std::hex << std::setw(16)
              << std::setfill('0') << bits << std::dec << '\n';
}

void printEvaluation(std::string_view grouping, const Evaluation& evaluation) {
    for (int row = 0; row < 2; ++row) {
        printScalar(grouping, "r", row, -1, evaluation.residuals[row].a);
    }
    // Match Ceres CostFunction storage: one residual-major 2-column matrix
    // per parameter block.
    for (int block = 0; block < 2; ++block) {
        for (int row = 0; row < 2; ++row) {
            for (int local = 0; local < 2; ++local) {
                printScalar(grouping, "J", row, block * 2 + local,
                            evaluation.residuals[row].v[block * 2 + local]);
            }
        }
    }
}

}  // namespace

int main() {
    printEvaluation("left", evaluate(false));
    printEvaluation("square_first", evaluate(true));
    return 0;
}
