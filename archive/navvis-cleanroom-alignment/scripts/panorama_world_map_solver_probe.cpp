#include <Eigen/Core>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using Vector = Eigen::VectorXd;

constexpr double kDataScale = 0.05;
constexpr double kInnerSquaredResidualTolerance = 1.0e-15;
constexpr double kOuterRelativeObjectiveTolerance = 1.0e-6;
constexpr int kMaximumInnerIterations = 10000;
constexpr int kMaximumOuterIterations = 100;

struct LevelShape {
    int width;
    int height;
};

struct LinearSystem {
    int width;
    int height;
    double ray_weight;
    Vector measured;
    std::vector<std::uint8_t> valid;
    Vector inverse_diagonal;
};

struct SolverTrace {
    int outer_iterations = 0;
    int inner_iterations = 0;
    double initial_objective = 0.0;
    double final_objective = 0.0;
    double final_squared_residual = 0.0;
};

template <typename Value>
std::vector<Value> readRaw(const std::filesystem::path& path, std::size_t count) {
    std::vector<Value> values(count);
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        throw std::runtime_error("Cannot open input: " + path.string());
    }
    input.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(Value)));
    if (!input || input.peek() != std::ifstream::traits_type::eof()) {
        throw std::runtime_error("Unexpected input size: " + path.string());
    }
    return values;
}

void writeRaw(const std::filesystem::path& path, const Vector& values) {
    std::ofstream output(path, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error("Cannot open output: " + path.string());
    }
    output.write(
        reinterpret_cast<const char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(double)));
    if (!output) {
        throw std::runtime_error("Cannot write output: " + path.string());
    }
}

std::uint64_t bits(double value) {
    std::uint64_t result = 0;
    static_assert(sizeof(result) == sizeof(value));
    std::memcpy(&result, &value, sizeof(result));
    return result;
}

std::uint64_t orderedBits(double value) {
    const std::uint64_t raw = bits(value);
    return (raw & (std::uint64_t{1} << 63U)) != 0U
               ? ~raw + std::uint64_t{1}
               : raw | (std::uint64_t{1} << 63U);
}

std::uint64_t ulpDistance(double first, double second) {
    const std::uint64_t a = orderedBits(first);
    const std::uint64_t b = orderedBits(second);
    return a >= b ? a - b : b - a;
}

std::size_t indexOf(int row, int column, int width) {
    return static_cast<std::size_t>(row) * static_cast<std::size_t>(width) +
           static_cast<std::size_t>(column);
}

LinearSystem makeSystem(
    int width, int height, const std::vector<double>& measured,
    const std::vector<std::uint8_t>& valid, double ray_weight) {
    const Eigen::Index count = static_cast<Eigen::Index>(measured.size());
    LinearSystem system{
        width,
        height,
        ray_weight,
        Eigen::Map<const Vector>(measured.data(), count),
        valid,
        Vector::Zero(count),
    };

    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const std::size_t index = indexOf(row, column, width);
            // Match the installed scalar instruction order.  The data term is
            // accumulated before the vertical neighbours and the constant
            // horizontal degree; forming 2 + degree first is algebraically
            // equivalent but changes the reciprocal by up to two ULPs.
            double diagonal = 0.0;
            if (system.valid[index] != 0U) {
                diagonal = ray_weight;
                diagonal *= kDataScale;
                diagonal *= system.measured[index];
                diagonal += 0.0;
            }
            if (row != 0) {
                diagonal += 1.0;
            }
            if (row + 1 != height) {
                diagonal += 1.0;
            }
            diagonal += 2.0;
            system.inverse_diagonal[static_cast<Eigen::Index>(index)] =
                diagonal > 1.0e-6 ? 1.0 / diagonal : 1.0;
        }
    }
    return system;
}

// The installed binary forms -gradient directly.  In particular, the data
// residual is evaluated as ((ray_weight * 0.05) * measured) *
// (estimate - measured).  Replacing this with H*x-b changes the rounding
// before the first PCG iteration.
Vector negativeGradient(const LinearSystem& system, const Vector& estimate) {
    Vector residual(estimate.size());
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const std::size_t index = indexOf(row, column, system.width);
            const Eigen::Index i = static_cast<Eigen::Index>(index);
            const double current = estimate[i];

            double non_horizontal = 0.0;
            if (system.valid[index] != 0U) {
                non_horizontal = system.ray_weight;
                non_horizontal *= kDataScale;
                non_horizontal *= system.measured[i];
                non_horizontal *= current - system.measured[i];
                non_horizontal += 0.0;
            }
            if (row != 0) {
                non_horizontal +=
                    current - estimate[i - static_cast<Eigen::Index>(system.width)];
            }
            if (row + 1 != system.height) {
                non_horizontal +=
                    current + estimate[i + static_cast<Eigen::Index>(system.width)] * -1.0;
            }

            const int left_column = column == 0 ? system.width - 1 : column - 1;
            const int right_column = column + 1 == system.width ? 0 : column + 1;
            double horizontal = current + current;
            horizontal -= estimate[static_cast<Eigen::Index>(
                indexOf(row, right_column, system.width))];
            horizontal -= estimate[static_cast<Eigen::Index>(
                indexOf(row, left_column, system.width))];
            horizontal += non_horizontal;
            residual[i] = -horizontal;
        }
    }
    return residual;
}

Vector applyHessian(const LinearSystem& system, const Vector& input) {
    Vector output(input.size());
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const std::size_t index = indexOf(row, column, system.width);
            const Eigen::Index i = static_cast<Eigen::Index>(index);
            const double current = input[i];

            double non_horizontal = 0.0;
            if (system.valid[index] != 0U) {
                non_horizontal = system.ray_weight;
                non_horizontal *= kDataScale;
                non_horizontal *= system.measured[i];
                non_horizontal *= current;
                non_horizontal += 0.0;
            }
            if (row != 0) {
                non_horizontal +=
                    current - input[i - static_cast<Eigen::Index>(system.width)];
            }
            if (row + 1 != system.height) {
                non_horizontal +=
                    current + input[i + static_cast<Eigen::Index>(system.width)] * -1.0;
            }

            const int left_column = column == 0 ? system.width - 1 : column - 1;
            const int right_column = column + 1 == system.width ? 0 : column + 1;
            double horizontal = current + current;
            horizontal -= input[static_cast<Eigen::Index>(
                indexOf(row, right_column, system.width))];
            horizontal -= input[static_cast<Eigen::Index>(
                indexOf(row, left_column, system.width))];
            output[i] = horizontal + non_horizontal;
        }
    }
    return output;
}

double objective(const LinearSystem& system, const Vector& estimate) {
    double result = 0.0;
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const std::size_t index = indexOf(row, column, system.width);
            const Eigen::Index i = static_cast<Eigen::Index>(index);
            const double current = estimate[i];
            if (system.valid[index] != 0U) {
                const double difference = current - system.measured[i];
                result += 0.5 * system.ray_weight * kDataScale *
                          system.measured[i] * difference * difference;
            }

            const int right_column = column + 1 == system.width ? 0 : column + 1;
            const double horizontal_difference =
                current - estimate[static_cast<Eigen::Index>(
                              indexOf(row, right_column, system.width))];
            result += 0.5 * horizontal_difference * horizontal_difference;
            if (row + 1 != system.height) {
                const double vertical_difference =
                    current - estimate[i + static_cast<Eigen::Index>(system.width)];
                result += 0.5 * vertical_difference * vertical_difference;
            }
        }
    }
    return result;
}

Vector solveLevel(
    const LinearSystem& system, const Vector& initial, bool zero_initial,
    const std::filesystem::path& trace_directory, int level,
    SolverTrace& trace) {
    Vector estimate = initial;
    if (zero_initial) {
        estimate.setZero();
    }

    double previous_objective = objective(system, estimate);
    trace.initial_objective = previous_objective;
    trace.final_objective = previous_objective;

    for (int outer = 0; outer < kMaximumOuterIterations; ++outer) {
        Vector residual = negativeGradient(system, estimate);
        double squared_residual = residual.squaredNorm();
        trace.final_squared_residual = squared_residual;
        if (kInnerSquaredResidualTolerance > squared_residual) {
            break;
        }

        Vector preconditioned = residual.cwiseProduct(system.inverse_diagonal);
        Vector direction = preconditioned;
        double residual_product = preconditioned.dot(residual);
        Vector delta = Vector::Zero(estimate.size());
        if (outer == 0) {
            const std::string prefix =
                "level" + std::to_string(level) + "_outer0_inner0_gradient_ready_";
            writeRaw(trace_directory / (prefix + "residual.raw"), residual);
            writeRaw(
                trace_directory / (prefix + "inverse_diagonal.raw"),
                system.inverse_diagonal);
            writeRaw(
                trace_directory / (prefix + "preconditioned.raw"), preconditioned);
            writeRaw(trace_directory / (prefix + "direction.raw"), direction);
            writeRaw(trace_directory / (prefix + "delta.raw"), delta);
        }

        for (int inner = 0; inner < kMaximumInnerIterations; ++inner) {
            const Vector hessian_times_direction = applyHessian(system, direction);
            const double denominator = hessian_times_direction.dot(direction);
            const double step = residual_product / denominator;
            delta += step * direction;
            residual -= step * hessian_times_direction;
            ++trace.inner_iterations;

            if (outer == 0 && inner == 0) {
                const std::string prefix =
                    "level" + std::to_string(level) +
                    "_outer0_inner0_first_step_ready_";
                writeRaw(
                    trace_directory / (prefix + "hessian_direction.raw"),
                    hessian_times_direction);
                writeRaw(trace_directory / (prefix + "residual.raw"), residual);
                writeRaw(trace_directory / (prefix + "direction.raw"), direction);
                writeRaw(trace_directory / (prefix + "delta.raw"), delta);
                writeRaw(
                    trace_directory / (prefix + "preconditioned.raw"),
                    preconditioned);
                writeRaw(
                    trace_directory / (prefix + "inverse_diagonal.raw"),
                    system.inverse_diagonal);
            }

            squared_residual = residual.squaredNorm();
            trace.final_squared_residual = squared_residual;
            if (kInnerSquaredResidualTolerance > squared_residual) {
                break;
            }

            preconditioned = residual.cwiseProduct(system.inverse_diagonal);
            const double next_product = preconditioned.dot(residual);
            const double beta = next_product / residual_product;
            direction = preconditioned + beta * direction;
            residual_product = next_product;
        }

        estimate += delta;
        ++trace.outer_iterations;
        const double next_objective = objective(system, estimate);
        trace.final_objective = next_objective;
        double relative_change = std::abs(next_objective - previous_objective);
        if (previous_objective > 0.0) {
            relative_change /= previous_objective;
        }
        if (kOuterRelativeObjectiveTolerance > relative_change) {
            break;
        }
        previous_objective = next_objective;
    }
    return estimate;
}

void compare(
    int level, const Vector& clean, const std::vector<double>& reference) {
    if (clean.size() != static_cast<Eigen::Index>(reference.size())) {
        throw std::runtime_error("Reference size mismatch");
    }

    std::size_t exact = 0;
    std::size_t first_difference = reference.size();
    long double absolute_sum = 0.0L;
    long double squared_sum = 0.0L;
    double maximum_absolute = 0.0;
    std::uint64_t maximum_ulp = 0;
    for (std::size_t index = 0; index < reference.size(); ++index) {
        const double value = clean[static_cast<Eigen::Index>(index)];
        const double target = reference[index];
        if (bits(value) == bits(target)) {
            ++exact;
        } else if (first_difference == reference.size()) {
            first_difference = index;
        }
        const double absolute = std::abs(value - target);
        absolute_sum += absolute;
        squared_sum += static_cast<long double>(absolute) * absolute;
        maximum_absolute = std::max(maximum_absolute, absolute);
        maximum_ulp = std::max(maximum_ulp, ulpDistance(value, target));
    }

    const long double count = static_cast<long double>(reference.size());
    std::cout << "level=" << level
              << " exact=" << exact << '/' << reference.size()
              << " mae=" << static_cast<double>(absolute_sum / count)
              << " rmse=" << std::sqrt(static_cast<double>(squared_sum / count))
              << " max_abs=" << maximum_absolute
              << " max_ulp=" << maximum_ulp;
    if (first_difference != reference.size()) {
        const double value = clean[static_cast<Eigen::Index>(first_difference)];
        const double target = reference[first_difference];
        std::cout << " first_index=" << first_difference
                  << " first_clean_bits=0x" << std::hex << bits(value)
                  << " first_vendor_bits=0x" << bits(target) << std::dec
                  << " first_delta=" << (value - target)
                  << " first_ulp=" << ulpDistance(value, target);
    }
    std::cout << '\n';
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc != 3) {
            std::cerr <<
                "Usage: panorama_world_map_solver_probe DEPTH_SOLVER_DUMP OUTPUT_DIR\n";
            return 2;
        }

        const std::filesystem::path input_directory = argv[1];
        const std::filesystem::path output_directory = argv[2];
        std::filesystem::create_directories(output_directory);
        const std::vector<LevelShape> shapes{
            {128, 64}, {256, 128}, {512, 256}, {1024, 512}};

        std::cout << std::setprecision(17);
        for (std::size_t level_index = 0; level_index < shapes.size(); ++level_index) {
            const int level = static_cast<int>(level_index + 1U);
            const LevelShape shape = shapes[level_index];
            const std::size_t count =
                static_cast<std::size_t>(shape.width) * shape.height;
            const std::string prefix = "level" + std::to_string(level);
            const std::vector<double> measured = readRaw<double>(
                input_directory / (prefix + "_measured.raw"), count);
            const std::vector<std::uint8_t> mask = readRaw<std::uint8_t>(
                input_directory / (prefix + "_mask.raw"), count);
            const std::vector<double> reference = readRaw<double>(
                input_directory / (prefix + "_estimate.raw"), count);

            const LinearSystem system =
                makeSystem(shape.width, shape.height, measured, mask, 1.0);
            const Vector initial = Eigen::Map<const Vector>(
                measured.data(), static_cast<Eigen::Index>(count));
            SolverTrace trace;
            const Vector clean = solveLevel(
                system, initial, level == 1, output_directory, level, trace);
            writeRaw(output_directory / (prefix + "_estimate_clean.raw"), clean);

            std::cout << "level=" << level
                      << " outer=" << trace.outer_iterations
                      << " inner=" << trace.inner_iterations
                      << " objective_initial=" << trace.initial_objective
                      << " objective_final=" << trace.final_objective
                      << " residual2=" << trace.final_squared_residual << '\n';
            compare(level, clean, reference);
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "ERROR: " << error.what() << '\n';
        return 1;
    }
}
