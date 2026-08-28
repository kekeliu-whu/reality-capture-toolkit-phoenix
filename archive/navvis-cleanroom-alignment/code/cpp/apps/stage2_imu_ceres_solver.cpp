// Standalone Ceres worker for the clean-room SurveyorSLAM Stage2 problem.
//
// Command line:
//   navvis_recon_stage2_imu_ceres_solver INPUT.bin OUTPUT.bin
//
// Both files are versioned packed little-endian streams. Quaternions use
// (x, y, z, w) storage. The input contains only the Stage1 state, graph,
// raw IMU stream, calibration initial values and public clean-room options.
// A vendor Stage2 result is never an input to this worker.

#include <ceres/version.h>
#if CERES_VERSION_MAJOR > 2 || \
    (CERES_VERSION_MAJOR == 2 && CERES_VERSION_MINOR >= 2)
#include <ceres/autodiff_manifold.h>
#define NAVVIS_RECON_CERES_USES_MANIFOLD 1
#else
#include <ceres/autodiff_local_parameterization.h>
#define NAVVIS_RECON_CERES_USES_MANIFOLD 0
#endif
#include <ceres/ceres.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace {

constexpr std::array<char, 8> kInputMagic = {'N', 'V', 'S', 'G', '2', 'C', 'R', '1'};
constexpr std::array<char, 8> kOutputMagic = {'N', 'V', 'S', 'G', '2', 'R', 'S', '1'};
constexpr std::uint32_t kSchemaVersion = 1;
constexpr double kLoopClosureHuberScale = 2500.0;

struct Pose {
    std::array<double, 3> translation{};
    std::array<double, 4> rotation_xyzw{};
};

struct Edge {
    std::uint32_t source = 0;
    std::uint32_t target = 0;
    std::uint32_t kind = 0;
    std::array<double, 3> measured_translation{};
    std::array<double, 4> measured_rotation_xyzw{};
    double translation_weight = 0.0;
    double rotation_weight = 0.0;
};

struct RawImuSample {
    std::array<double, 3> linear_acceleration{};
    std::array<double, 3> angular_velocity{};
};

struct IntegrationPoint {
    std::uint32_t before_sample = 0;
    std::uint32_t after_sample = 0;
    double alpha = 0.0;
    double dt_to_next = 0.0;
};

struct ImuFactor {
    std::uint32_t source_vertex = 0;
    std::uint32_t target_vertex = 0;
    std::uint32_t source_velocity = 0;
    std::uint32_t target_velocity = 0;
    double duration = 0.0;
    std::vector<IntegrationPoint> points;
};

struct Calibration {
    double gravity = 0.0;
    std::array<double, 4> imu_rotation_xyzw{};
    std::array<double, 3> linear_acceleration_bias{};
    std::array<double, 3> linear_acceleration_scaling{};
    std::array<double, 6> linear_acceleration_cross_axis{};
    std::array<double, 3> angular_velocity_bias{};
    std::array<double, 3> angular_velocity_scaling{};
    std::array<double, 6> angular_velocity_cross_axis{};
};

struct CalibrationOptions {
    double gravity_target = 0.0;
    double gravity_weight = 0.0;
    double imu_orientation_weight = 0.0;
    std::array<double, 3> linear_acceleration_bias_weight{};
    std::array<double, 3> linear_acceleration_scaling_weight{};
    std::array<double, 3> angular_velocity_bias_weight{};
    std::array<double, 3> angular_velocity_scaling_weight{};
};

struct InputProblem {
    std::uint32_t fixed_vertex = 0;
    std::uint32_t max_iterations = 0;
    std::uint32_t num_threads = 0;
    double imu_rotation_weight = 0.0;
    double imu_velocity_weight = 0.0;
    double imu_position_weight = 0.0;
    Calibration calibration;
    CalibrationOptions calibration_options;
    std::vector<Pose> poses;
    std::vector<std::uint32_t> node_vertices;
    std::vector<std::array<double, 3>> velocities;
    std::vector<Edge> edges;
    std::vector<RawImuSample> samples;
    std::vector<ImuFactor> imu_factors;
};

class LittleEndianReader {
public:
    explicit LittleEndianReader(const std::string& path) {
        std::ifstream stream(path, std::ios::binary | std::ios::ate);
        if (!stream) {
            throw std::runtime_error("cannot open input file: " + path);
        }
        const std::streamoff end = stream.tellg();
        if (end < 0 || static_cast<std::uintmax_t>(end) >
                           std::numeric_limits<std::size_t>::max()) {
            throw std::runtime_error("input file size is not representable");
        }
        data_.resize(static_cast<std::size_t>(end));
        stream.seekg(0, std::ios::beg);
        if (!data_.empty()) {
            stream.read(reinterpret_cast<char*>(data_.data()),
                        static_cast<std::streamsize>(data_.size()));
        }
        if (!stream) {
            throw std::runtime_error("failed to read complete input file: " + path);
        }
    }

    std::array<char, 8> readMagic() {
        require(8);
        std::array<char, 8> value{};
        std::memcpy(value.data(), data_.data() + offset_, value.size());
        offset_ += value.size();
        return value;
    }

    std::uint32_t readUint32() {
        require(4);
        std::uint32_t value = 0;
        for (unsigned int byte = 0; byte < 4; ++byte) {
            value |= static_cast<std::uint32_t>(data_[offset_ + byte]) << (8U * byte);
        }
        offset_ += 4;
        return value;
    }

    double readDouble() {
        require(8);
        std::uint64_t bits = 0;
        for (unsigned int byte = 0; byte < 8; ++byte) {
            bits |= static_cast<std::uint64_t>(data_[offset_ + byte]) << (8U * byte);
        }
        offset_ += 8;
        double value = 0.0;
        static_assert(sizeof(value) == sizeof(bits), "binary64 double required");
        std::memcpy(&value, &bits, sizeof(value));
        return value;
    }

    std::size_t size() const { return data_.size(); }
    std::size_t offset() const { return offset_; }

private:
    void require(const std::size_t bytes) const {
        if (bytes > data_.size() - offset_) {
            throw std::runtime_error("truncated input at byte " + std::to_string(offset_));
        }
    }

    std::vector<std::uint8_t> data_;
    std::size_t offset_ = 0;
};

class LittleEndianWriter {
public:
    void writeMagic(const std::array<char, 8>& magic) {
        data_.insert(data_.end(), magic.begin(), magic.end());
    }

    void writeUint32(const std::uint32_t value) {
        for (unsigned int byte = 0; byte < 4; ++byte) {
            data_.push_back(static_cast<std::uint8_t>((value >> (8U * byte)) & 0xffU));
        }
    }

    void writeDouble(const double value) {
        std::uint64_t bits = 0;
        static_assert(sizeof(value) == sizeof(bits), "binary64 double required");
        std::memcpy(&bits, &value, sizeof(bits));
        for (unsigned int byte = 0; byte < 8; ++byte) {
            data_.push_back(static_cast<std::uint8_t>((bits >> (8U * byte)) & 0xffU));
        }
    }

    void save(const std::string& path) const {
        std::ofstream stream(path, std::ios::binary | std::ios::trunc);
        if (!stream) {
            throw std::runtime_error("cannot open output file: " + path);
        }
        if (!data_.empty()) {
            stream.write(reinterpret_cast<const char*>(data_.data()),
                         static_cast<std::streamsize>(data_.size()));
        }
        if (!stream) {
            throw std::runtime_error("failed to write complete output file: " + path);
        }
    }

private:
    std::vector<std::uint8_t> data_;
};

template <std::size_t Size>
std::array<double, Size> readDoubleArray(LittleEndianReader& reader) {
    std::array<double, Size> values{};
    for (double& value : values) {
        value = reader.readDouble();
    }
    return values;
}

template <std::size_t Size>
void requireFinite(const std::array<double, Size>& values, const std::string& name) {
    for (const double value : values) {
        if (!std::isfinite(value)) {
            throw std::runtime_error(name + " contains a non-finite value");
        }
    }
}

void requireFinite(const double value, const std::string& name) {
    if (!std::isfinite(value)) {
        throw std::runtime_error(name + " is not finite");
    }
}

void requirePositive(const double value, const std::string& name) {
    requireFinite(value, name);
    if (!(value > 0.0)) {
        throw std::runtime_error(name + " must be positive");
    }
}

void requireNonnegative(const double value, const std::string& name) {
    requireFinite(value, name);
    if (value < 0.0) {
        throw std::runtime_error(name + " must be non-negative");
    }
}

double quaternionSquaredNorm(const std::array<double, 4>& quaternion,
                             const std::string& name) {
    requireFinite(quaternion, name);
    double squared_norm = 0.0;
    for (const double value : quaternion) {
        squared_norm += value * value;
    }
    if (!(squared_norm > std::numeric_limits<double>::min())) {
        throw std::runtime_error(name + " has zero norm");
    }
    return squared_norm;
}

void normalizeQuaternion(std::array<double, 4>& quaternion, const std::string& name) {
    const double squared_norm = quaternionSquaredNorm(quaternion, name);
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (double& value : quaternion) {
        value *= inverse_norm;
    }
}

InputProblem readInput(const std::string& path) {
    LittleEndianReader reader(path);
    if (reader.readMagic() != kInputMagic) {
        throw std::runtime_error("input magic is not NVSG2CR1");
    }
    if (reader.readUint32() != kSchemaVersion) {
        throw std::runtime_error("unsupported Stage2 input schema version");
    }

    const std::uint32_t pose_count = reader.readUint32();
    const std::uint32_t node_count = reader.readUint32();
    const std::uint32_t edge_count = reader.readUint32();
    const std::uint32_t sample_count = reader.readUint32();
    const std::uint32_t factor_count = reader.readUint32();

    InputProblem input;
    input.fixed_vertex = reader.readUint32();
    input.max_iterations = reader.readUint32();
    input.num_threads = reader.readUint32();
    const std::uint32_t reserved = reader.readUint32();
    if (reserved != 0) {
        throw std::runtime_error("Stage2 input reserved field must be zero");
    }

    input.imu_rotation_weight = reader.readDouble();
    input.imu_velocity_weight = reader.readDouble();
    input.imu_position_weight = reader.readDouble();
    input.calibration.gravity = reader.readDouble();
    input.calibration.imu_rotation_xyzw = readDoubleArray<4>(reader);
    input.calibration.linear_acceleration_bias = readDoubleArray<3>(reader);
    input.calibration.linear_acceleration_scaling = readDoubleArray<3>(reader);
    input.calibration.linear_acceleration_cross_axis = readDoubleArray<6>(reader);
    input.calibration.angular_velocity_bias = readDoubleArray<3>(reader);
    input.calibration.angular_velocity_scaling = readDoubleArray<3>(reader);
    input.calibration.angular_velocity_cross_axis = readDoubleArray<6>(reader);
    input.calibration_options.gravity_target = reader.readDouble();
    input.calibration_options.gravity_weight = reader.readDouble();
    input.calibration_options.imu_orientation_weight = reader.readDouble();
    input.calibration_options.linear_acceleration_bias_weight =
        readDoubleArray<3>(reader);
    input.calibration_options.linear_acceleration_scaling_weight =
        readDoubleArray<3>(reader);
    input.calibration_options.angular_velocity_bias_weight =
        readDoubleArray<3>(reader);
    input.calibration_options.angular_velocity_scaling_weight =
        readDoubleArray<3>(reader);

    if (pose_count == 0 || node_count < 2 || node_count > pose_count ||
        factor_count != node_count - 1 || sample_count < 2) {
        throw std::runtime_error("Stage2 record counts are inconsistent");
    }
    if (input.fixed_vertex >= pose_count || input.max_iterations == 0 ||
        input.num_threads == 0) {
        throw std::runtime_error("Stage2 gauge or solver options are invalid");
    }
    requirePositive(input.imu_rotation_weight, "IMU rotation weight");
    requirePositive(input.imu_velocity_weight, "IMU velocity weight");
    requirePositive(input.imu_position_weight, "IMU position weight");
    requirePositive(input.calibration.gravity, "gravity");
    normalizeQuaternion(input.calibration.imu_rotation_xyzw, "IMU orientation");
    requireFinite(input.calibration.linear_acceleration_bias, "acceleration bias");
    requireFinite(input.calibration.linear_acceleration_scaling,
                  "acceleration scaling");
    requireFinite(input.calibration.linear_acceleration_cross_axis,
                  "acceleration cross axis");
    requireFinite(input.calibration.angular_velocity_bias, "gyro bias");
    requireFinite(input.calibration.angular_velocity_scaling, "gyro scaling");
    requireFinite(input.calibration.angular_velocity_cross_axis, "gyro cross axis");
    requirePositive(input.calibration_options.gravity_target, "gravity prior target");
    requireNonnegative(input.calibration_options.gravity_weight, "gravity prior weight");
    requireNonnegative(input.calibration_options.imu_orientation_weight,
                       "IMU orientation prior weight");
    for (const auto* weights : {
             &input.calibration_options.linear_acceleration_bias_weight,
             &input.calibration_options.linear_acceleration_scaling_weight,
             &input.calibration_options.angular_velocity_bias_weight,
             &input.calibration_options.angular_velocity_scaling_weight}) {
        for (const double weight : *weights) {
            requireNonnegative(weight, "intrinsic prior weight");
        }
    }

    input.poses.resize(pose_count);
    for (Pose& pose : input.poses) {
        pose.translation = readDoubleArray<3>(reader);
        pose.rotation_xyzw = readDoubleArray<4>(reader);
        requireFinite(pose.translation, "pose translation");
        // The installed Stage2 factor consumes the serialized coefficients
        // directly on its first evaluation. Some pre-intrinsics poses are a
        // few ulps away from unit length, so normalizing here changes the
        // initial residual and the three-step Ceres trajectory.
        quaternionSquaredNorm(pose.rotation_xyzw, "pose rotation");
    }

    input.node_vertices.resize(node_count);
    input.velocities.resize(node_count);
    std::vector<bool> vertex_seen(pose_count, false);
    for (std::uint32_t index = 0; index < node_count; ++index) {
        input.node_vertices[index] = reader.readUint32();
        input.velocities[index] = readDoubleArray<3>(reader);
        const std::uint32_t vertex = input.node_vertices[index];
        if (vertex >= pose_count || vertex_seen[vertex]) {
            throw std::runtime_error("node vertex list is invalid or duplicated");
        }
        vertex_seen[vertex] = true;
        requireFinite(input.velocities[index], "node velocity");
    }

    input.edges.resize(edge_count);
    for (Edge& edge : input.edges) {
        edge.source = reader.readUint32();
        edge.target = reader.readUint32();
        edge.kind = reader.readUint32();
        edge.measured_translation = readDoubleArray<3>(reader);
        edge.measured_rotation_xyzw = readDoubleArray<4>(reader);
        edge.translation_weight = reader.readDouble();
        edge.rotation_weight = reader.readDouble();
        if (edge.source >= pose_count || edge.target >= pose_count || edge.kind > 1) {
            throw std::runtime_error("graph edge has an invalid vertex or kind");
        }
        requireFinite(edge.measured_translation, "graph edge translation");
        // Keep the exact frontend coefficients used by the installed graph
        // residual instead of replacing them with an equivalent unit value.
        quaternionSquaredNorm(edge.measured_rotation_xyzw, "graph edge rotation");
        requirePositive(edge.translation_weight, "graph translation weight");
        requirePositive(edge.rotation_weight, "graph rotation weight");
    }

    input.samples.resize(sample_count);
    for (RawImuSample& sample : input.samples) {
        sample.linear_acceleration = readDoubleArray<3>(reader);
        sample.angular_velocity = readDoubleArray<3>(reader);
        requireFinite(sample.linear_acceleration, "raw acceleration");
        requireFinite(sample.angular_velocity, "raw angular velocity");
    }

    input.imu_factors.resize(factor_count);
    for (ImuFactor& factor : input.imu_factors) {
        factor.source_vertex = reader.readUint32();
        factor.target_vertex = reader.readUint32();
        factor.source_velocity = reader.readUint32();
        factor.target_velocity = reader.readUint32();
        const std::uint32_t point_count = reader.readUint32();
        factor.duration = reader.readDouble();
        if (factor.source_vertex >= pose_count || factor.target_vertex >= pose_count ||
            factor.source_velocity >= node_count || factor.target_velocity >= node_count ||
            point_count < 2) {
            throw std::runtime_error("IMU factor indices are invalid");
        }
        requirePositive(factor.duration, "IMU factor duration");
        factor.points.resize(point_count);
        double integrated_duration = 0.0;
        for (std::uint32_t point_index = 0; point_index < point_count; ++point_index) {
            IntegrationPoint& point = factor.points[point_index];
            point.before_sample = reader.readUint32();
            point.after_sample = reader.readUint32();
            point.alpha = reader.readDouble();
            point.dt_to_next = reader.readDouble();
            if (point.before_sample >= sample_count || point.after_sample >= sample_count ||
                point.before_sample > point.after_sample || point.alpha < 0.0 ||
                point.alpha > 1.0 || point.dt_to_next < 0.0 ||
                !std::isfinite(point.alpha) || !std::isfinite(point.dt_to_next)) {
                throw std::runtime_error("IMU integration point is invalid");
            }
            if (point_index + 1 == point_count && point.dt_to_next != 0.0) {
                throw std::runtime_error("last IMU integration point has nonzero dt");
            }
            integrated_duration += point.dt_to_next;
        }
        if (std::abs(integrated_duration - factor.duration) > 1.0e-12) {
            throw std::runtime_error("IMU integration points do not span factor duration");
        }
    }
    if (reader.offset() != reader.size()) {
        throw std::runtime_error("Stage2 input contains trailing bytes");
    }
    return input;
}

template <typename T>
Eigen::Quaternion<T> quaternionFromXyzw(const T* values) {
    return Eigen::Quaternion<T>(values[3], values[0], values[1], values[2]);
}

template <typename T>
Eigen::Quaternion<T> quaternionFromXyzw(const std::array<double, 4>& values) {
    return Eigen::Quaternion<T>(T(values[3]), T(values[0]), T(values[1]), T(values[2]));
}

template <typename T>
Eigen::Quaternion<T> quaternionFromRotationVector(const Eigen::Matrix<T, 3, 1>& vector) {
    const T angle = ceres::sqrt(vector.squaredNorm());
    const T half_angle = T(0.5) * angle;
    const T imaginary_scale =
        angle < T(1.0e-10) ? T(0.5) : ceres::sin(half_angle) / angle;
    return Eigen::Quaternion<T>(ceres::cos(half_angle), imaginary_scale * vector.x(),
                                imaginary_scale * vector.y(),
                                imaginary_scale * vector.z());
}

template <typename T>
void writeRotationVectorResidual(Eigen::Quaternion<T> error, const double weight,
                                 T* residual) {
    if (error.w() < T(0.0)) {
        error.coeffs() *= T(-1.0);
    }
    const T sine_half_angle = ceres::sqrt(error.vec().squaredNorm());
    const T vector_scale = sine_half_angle < T(1.0e-10)
                               ? T(2.0)
                               : T(2.0) * ceres::atan2(sine_half_angle, error.w()) /
                                     sine_half_angle;
    residual[0] = T(weight) * vector_scale * error.x();
    residual[1] = T(weight) * vector_scale * error.y();
    residual[2] = T(weight) * vector_scale * error.z();
}

// The Stage2 gauge keeps the first Submap translation and yaw fixed while
// allowing a right-multiplied roll/pitch correction.
struct ConstantYawQuaternionPlus {
    template <typename T>
    bool operator()(const T* x, const T* delta, T* x_plus_delta) const {
        return Plus(x, delta, x_plus_delta);
    }

    template <typename T>
    bool Plus(const T* x, const T* delta, T* x_plus_delta) const {
        const T delta_norm = ceres::sqrt(delta[0] * delta[0] + delta[1] * delta[1]);
        const T sin_delta_over_delta =
            delta_norm < T(1.0e-6) ? T(1.0) : ceres::sin(delta_norm) / delta_norm;
        const T cos_delta =
            delta_norm < T(1.0e-6) ? T(1.0) : ceres::cos(delta_norm);
        const Eigen::Quaternion<T> current = quaternionFromXyzw(x);
        const Eigen::Quaternion<T> increment(cos_delta,
                                              sin_delta_over_delta * delta[0],
                                              sin_delta_over_delta * delta[1], T(0.0));
        const Eigen::Quaternion<T> result = current * increment;
        x_plus_delta[0] = result.x();
        x_plus_delta[1] = result.y();
        x_plus_delta[2] = result.z();
        x_plus_delta[3] = result.w();
        return true;
    }

    template <typename T>
    bool Minus(const T* y, const T* x, T* y_minus_x) const {
        Eigen::Quaternion<T> difference =
            quaternionFromXyzw(x).conjugate() * quaternionFromXyzw(y);
        if (difference.w() < T(0.0)) {
            difference.coeffs() *= T(-1.0);
        }
        const T norm = ceres::sqrt(difference.x() * difference.x() +
                                   difference.y() * difference.y());
        const T scale = norm < T(1.0e-10)
                            ? T(1.0)
                            : ceres::atan2(norm, difference.w()) / norm;
        y_minus_x[0] = scale * difference.x();
        y_minus_x[1] = scale * difference.y();
        return true;
    }
};

struct GraphResidual {
    explicit GraphResidual(const Edge& edge)
        : measured_translation(edge.measured_translation),
          measured_rotation_xyzw(edge.measured_rotation_xyzw),
          translation_weight(edge.translation_weight),
          rotation_weight(edge.rotation_weight) {}

    template <typename T>
    bool operator()(const T* source_rotation, const T* source_translation,
                    const T* target_rotation, const T* target_translation,
                    T* residual) const {
        const Eigen::Quaternion<T> source = quaternionFromXyzw(source_rotation);
        const Eigen::Quaternion<T> target = quaternionFromXyzw(target_rotation);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> source_position(source_translation);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> target_position(target_translation);
        const Eigen::Matrix<T, 3, 1> predicted_translation =
            source.conjugate() * (target_position - source_position);
        for (int axis = 0; axis < 3; ++axis) {
            residual[axis] = T(translation_weight) *
                             (predicted_translation[axis] - T(measured_translation[axis]));
        }
        const Eigen::Quaternion<T> predicted_rotation = source.conjugate() * target;
        const Eigen::Quaternion<T> error =
            quaternionFromXyzw<T>(measured_rotation_xyzw).conjugate() *
            predicted_rotation;
        writeRotationVectorResidual(error, rotation_weight, residual + 3);
        return true;
    }

    std::array<double, 3> measured_translation;
    std::array<double, 4> measured_rotation_xyzw;
    double translation_weight;
    double rotation_weight;
};

struct ImuResidual {
    ImuResidual(const ImuFactor& factor, const std::vector<RawImuSample>& samples,
                const double rotation_weight, const double velocity_weight,
                const double position_weight)
        : duration(factor.duration),
          points(factor.points),
          samples(samples),
          rotation_scale(rotation_weight / std::sqrt(duration)),
          velocity_scale(velocity_weight / std::sqrt(duration)),
          position_scale(position_weight / std::sqrt(duration)) {}

    template <typename T>
    Eigen::Matrix<T, 3, 1> correctIntrinsic(
        const std::array<double, 3>& raw, const T* bias, const T* scaling,
        const T* cross_axis) const {
        Eigen::Matrix<T, 3, 1> intrinsic;
        for (int axis = 0; axis < 3; ++axis) {
            intrinsic[axis] = (T(raw[axis]) - bias[axis]) * scaling[axis];
        }
        Eigen::Matrix<T, 3, 1> misaligned;
        misaligned[0] = intrinsic[0] + cross_axis[0] * intrinsic[1] +
                        cross_axis[1] * intrinsic[2];
        misaligned[1] = cross_axis[2] * intrinsic[0] + intrinsic[1] +
                        cross_axis[3] * intrinsic[2];
        misaligned[2] = cross_axis[4] * intrinsic[0] +
                        cross_axis[5] * intrinsic[1] + intrinsic[2];
        return misaligned;
    }

    template <typename T>
    Eigen::Matrix<T, 3, 1> valueAt(
        const IntegrationPoint& point, const bool acceleration, const T* bias,
        const T* scaling, const T* cross_axis) const {
        const auto& before_raw = acceleration
                                     ? samples[point.before_sample].linear_acceleration
                                     : samples[point.before_sample].angular_velocity;
        const Eigen::Matrix<T, 3, 1> before =
            correctIntrinsic(before_raw, bias, scaling, cross_axis);
        if (point.before_sample == point.after_sample) {
            return before;
        }
        const auto& after_raw = acceleration
                                    ? samples[point.after_sample].linear_acceleration
                                    : samples[point.after_sample].angular_velocity;
        const Eigen::Matrix<T, 3, 1> after =
            correctIntrinsic(after_raw, bias, scaling, cross_axis);
        return T(1.0 - point.alpha) * before + T(point.alpha) * after;
    }

    template <typename T>
    bool operator()(const T* imu_rotation, const T* imu_translation,
                    const T* source_rotation, const T* source_translation,
                    const T* target_rotation, const T* target_translation,
                    const T* source_velocity, const T* target_velocity,
                    const T* angular_velocity_bias,
                    const T* angular_velocity_scaling,
                    const T* angular_velocity_cross_axis,
                    const T* acceleration_bias, const T* acceleration_scaling,
                    const T* acceleration_cross_axis, const T* gravity,
                    T* residual) const {
        // IMU translation is fixed to zero by this configuration.  Keeping it
        // in the signature reproduces the installed 15-block AutoDiff cost
        // function and therefore its Jet/Jacobian layout.
        (void)imu_translation;
        const Eigen::Quaternion<T> source = quaternionFromXyzw(source_rotation);
        const Eigen::Quaternion<T> target = quaternionFromXyzw(target_rotation);
        const Eigen::Quaternion<T> imu = quaternionFromXyzw(imu_rotation);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> source_position(source_translation);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> target_position(target_translation);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> source_velocity_map(source_velocity);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> target_velocity_map(target_velocity);

        Eigen::Quaternion<T> delta_rotation = Eigen::Quaternion<T>::Identity();
        Eigen::Matrix<T, 3, 1> delta_velocity = Eigen::Matrix<T, 3, 1>::Zero();
        Eigen::Matrix<T, 3, 1> delta_position = Eigen::Matrix<T, 3, 1>::Zero();
        const Eigen::Quaternion<T> source_imu = source * imu;
        const Eigen::Matrix<T, 3, 1> gravity_world(T(0.0), T(0.0), -gravity[0]);
        const Eigen::Matrix<T, 3, 1> gravity_imu =
            source_imu.inverse() * gravity_world;
        bool probe_steps = false;
        if constexpr (std::is_same_v<T, double>) {
            static bool steps_printed = false;
            probe_steps = !steps_printed &&
                          std::getenv("NAVVIS_RECON_STAGE2_PROBE_IMU_STEPS") != nullptr;
            steps_printed = steps_printed || probe_steps;
            if (probe_steps) {
                std::cerr << std::setprecision(17)
                          << "Stage2 IMU gravity_imu: " << gravity_imu.transpose()
                          << '\n';
            }
        }
        for (std::size_t index = 0; index + 1 < points.size(); ++index) {
            const IntegrationPoint& first_point = points[index];
            const IntegrationPoint& second_point = points[index + 1];
            const T dt = T(points[index].dt_to_next);

            std::uint32_t first_sample = first_point.before_sample;
            std::uint32_t second_sample = second_point.before_sample;
            double left_clip_ratio = 0.0;
            double right_clip_ratio = 0.0;
            if (first_point.before_sample != first_point.after_sample) {
                second_sample = first_point.after_sample;
                left_clip_ratio = first_point.alpha;
            } else if (second_point.before_sample != second_point.after_sample) {
                first_sample = second_point.before_sample;
                second_sample = second_point.after_sample;
                right_clip_ratio = 1.0 - second_point.alpha;
            }

            const Eigen::Matrix<T, 3, 1> raw_first_acceleration = correctIntrinsic(
                samples[first_sample].linear_acceleration, acceleration_bias,
                acceleration_scaling, acceleration_cross_axis);
            const Eigen::Matrix<T, 3, 1> raw_second_acceleration = correctIntrinsic(
                samples[second_sample].linear_acceleration, acceleration_bias,
                acceleration_scaling, acceleration_cross_axis);
            const Eigen::Matrix<T, 3, 1> raw_first_angular_velocity = correctIntrinsic(
                samples[first_sample].angular_velocity, angular_velocity_bias,
                angular_velocity_scaling, angular_velocity_cross_axis);
            const Eigen::Matrix<T, 3, 1> raw_second_angular_velocity = correctIntrinsic(
                samples[second_sample].angular_velocity, angular_velocity_bias,
                angular_velocity_scaling, angular_velocity_cross_axis);

            const Eigen::Matrix<T, 3, 1> first_angular_velocity =
                T(1.0 - left_clip_ratio) * raw_first_angular_velocity +
                T(left_clip_ratio) * raw_second_angular_velocity;
            const Eigen::Matrix<T, 3, 1> second_angular_velocity =
                T(right_clip_ratio) * raw_first_angular_velocity +
                T(1.0 - right_clip_ratio) * raw_second_angular_velocity;
            const Eigen::Matrix<T, 3, 1> delta_angle =
                T(0.5) * dt *
                (first_angular_velocity + second_angular_velocity);
            Eigen::Quaternion<T> next_delta_rotation =
                delta_rotation * quaternionFromRotationVector(delta_angle);
            next_delta_rotation.normalize();

            const Eigen::Matrix<T, 3, 1> rotated_first_acceleration =
                delta_rotation * raw_first_acceleration;
            const Eigen::Matrix<T, 3, 1> rotated_second_acceleration =
                next_delta_rotation * raw_second_acceleration;
            const Eigen::Matrix<T, 3, 1> first_acceleration =
                T(1.0 - left_clip_ratio) * rotated_first_acceleration +
                T(left_clip_ratio) * rotated_second_acceleration;
            const Eigen::Matrix<T, 3, 1> second_acceleration =
                T(right_clip_ratio) * rotated_first_acceleration +
                T(1.0 - right_clip_ratio) * rotated_second_acceleration;
            const Eigen::Matrix<T, 3, 1> acceleration_source =
                T(0.5) * (first_acceleration + second_acceleration) + gravity_imu;
            const Eigen::Matrix<T, 3, 1> previous_velocity = delta_velocity;
            delta_velocity += acceleration_source * dt;
            delta_position += T(0.5) * (previous_velocity + delta_velocity) * dt;
            delta_rotation = next_delta_rotation;
            if constexpr (std::is_same_v<T, double>) {
                if (probe_steps) {
                    std::cerr << "Stage2 IMU step " << index
                              << " delta_velocity=" << delta_velocity.transpose()
                              << " delta_position=" << delta_position.transpose()
                              << " delta_rotation_xyzw="
                              << delta_rotation.coeffs().transpose() << '\n';
                }
            }
        }

        const T dt = T(duration);
        delta_position += source_imu.inverse() * source_velocity_map * dt;
        const Eigen::Matrix<T, 3, 1> predicted_position =
            source_imu.inverse() * (target_position - source_position);
        for (int axis = 0; axis < 3; ++axis) {
            residual[axis] =
                T(position_scale) * (predicted_position[axis] - delta_position[axis]);
        }

        const Eigen::Quaternion<T> predicted_rotation =
            source_imu.inverse() * (target * imu);
        writeRotationVectorResidual(
            delta_rotation.conjugate() * predicted_rotation,
            rotation_scale, residual + 3);

        const Eigen::Matrix<T, 3, 1> predicted_velocity =
            target_velocity_map - source_velocity_map -
            source_imu * delta_velocity;
        for (int axis = 0; axis < 3; ++axis) {
            residual[6 + axis] = T(velocity_scale) * predicted_velocity[axis];
        }
        return true;
    }

    double duration;
    std::vector<IntegrationPoint> points;
    const std::vector<RawImuSample>& samples;
    double rotation_scale;
    double velocity_scale;
    double position_scale;
};

struct GravityPriorResidual {
    explicit GravityPriorResidual(const CalibrationOptions& options)
        : target(options.gravity_target), weight(options.gravity_weight) {}
    template <typename T>
    bool operator()(const T* gravity, T* residual) const {
        residual[0] = T(weight) * (gravity[0] - T(target));
        return true;
    }
    double target;
    double weight;
};

struct ImuOrientationPriorResidual {
    template <typename T>
    bool operator()(const T* rotation, T* residual) const {
        writeRotationVectorResidual(quaternionFromXyzw(rotation), 1.0, residual);
        return true;
    }
};

struct VectorPriorResidual {
    VectorPriorResidual(const std::array<double, 3>& target,
                        const std::array<double, 3>& weight)
        : target(target), weight(weight) {}
    template <typename T>
    bool operator()(const T* value, T* residual) const {
        for (int axis = 0; axis < 3; ++axis) {
            residual[axis] = T(weight[axis]) * (value[axis] - T(target[axis]));
        }
        return true;
    }
    std::array<double, 3> target;
    std::array<double, 3> weight;
};

struct SolveResult {
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::uint32_t iterations = 0;
    bool success = false;
};

SolveResult solve(InputProblem& input) {
    if (const char* value = std::getenv("NAVVIS_RECON_STAGE2_PROBE_IMU_FACTOR")) {
        char* end = nullptr;
        const unsigned long requested = std::strtoul(value, &end, 10);
        if (end == value || *end != '\0' || requested >= input.imu_factors.size()) {
            throw std::runtime_error(
                "NAVVIS_RECON_STAGE2_PROBE_IMU_FACTOR is outside the factor range");
        }
        const ImuFactor& factor = input.imu_factors[requested];
        const Pose& source = input.poses[factor.source_vertex];
        const Pose& target = input.poses[factor.target_vertex];
        ImuResidual probe_residual(factor, input.samples, input.imu_rotation_weight,
                                   input.imu_velocity_weight,
                                   input.imu_position_weight);
        std::cerr << "Stage2 IMU factor " << requested
                  << " duration=" << std::setprecision(17) << factor.duration
                  << " points=" << factor.points.size() << '\n';
        for (std::size_t point_index = 0; point_index < factor.points.size(); ++point_index) {
            const IntegrationPoint& point = factor.points[point_index];
            const Eigen::Vector3d acceleration = probe_residual.valueAt(
                point, true, input.calibration.linear_acceleration_bias.data(),
                input.calibration.linear_acceleration_scaling.data(),
                input.calibration.linear_acceleration_cross_axis.data());
            const Eigen::Vector3d angular_velocity = probe_residual.valueAt(
                point, false, input.calibration.angular_velocity_bias.data(),
                input.calibration.angular_velocity_scaling.data(),
                input.calibration.angular_velocity_cross_axis.data());
            std::cerr << "  point " << point_index << " before=" << point.before_sample
                      << " after=" << point.after_sample << " alpha=" << point.alpha
                      << " dt=" << point.dt_to_next << " acceleration="
                      << acceleration.transpose() << " angular_velocity="
                      << angular_velocity.transpose() << '\n';
        }
        std::array<double, 9> residual{};
        const std::array<double, 3> imu_translation = {0.0, 0.0, 0.0};
        probe_residual(
            input.calibration.imu_rotation_xyzw.data(), imu_translation.data(),
            source.rotation_xyzw.data(), source.translation.data(),
            target.rotation_xyzw.data(), target.translation.data(),
            input.velocities[factor.source_velocity].data(),
            input.velocities[factor.target_velocity].data(),
            input.calibration.angular_velocity_bias.data(),
            input.calibration.angular_velocity_scaling.data(),
            input.calibration.angular_velocity_cross_axis.data(),
            input.calibration.linear_acceleration_bias.data(),
            input.calibration.linear_acceleration_scaling.data(),
            input.calibration.linear_acceleration_cross_axis.data(),
            &input.calibration.gravity, residual.data());
        std::cerr << "Stage2 IMU factor " << requested << " initial residual:";
        std::cerr << std::setprecision(17);
        for (const double component : residual) {
            std::cerr << ' ' << component;
        }
        std::cerr << '\n';
    }

    ceres::Problem problem;
    const auto add_pose = [&problem, &input](const std::size_t index) {
        Pose& pose = input.poses[index];
        problem.AddParameterBlock(pose.translation.data(), 3);
        if (index == input.fixed_vertex) {
#if NAVVIS_RECON_CERES_USES_MANIFOLD
            problem.AddParameterBlock(
                pose.rotation_xyzw.data(), 4,
                new ceres::AutoDiffManifold<ConstantYawQuaternionPlus, 4, 2>());
#else
            problem.AddParameterBlock(
                pose.rotation_xyzw.data(), 4,
                new ceres::AutoDiffLocalParameterization<
                    ConstantYawQuaternionPlus, 4, 2>());
#endif
        } else {
#if NAVVIS_RECON_CERES_USES_MANIFOLD
            problem.AddParameterBlock(pose.rotation_xyzw.data(), 4,
                                      new ceres::EigenQuaternionManifold());
#else
            problem.AddParameterBlock(
                pose.rotation_xyzw.data(), 4,
                new ceres::EigenQuaternionParameterization());
#endif
        }
    };

    // The installed Stage2 program registers retained Submaps first.  It then
    // interleaves every node's translation, rotation and velocity blocks.
    // Ceres preserves this order when assembling the sparse system, so a
    // geometrically equivalent nodes-first/all-velocities-last layout changes
    // the low bits of the multi-threaded Cholesky result.
    std::vector<bool> is_node(input.poses.size(), false);
    for (const std::uint32_t vertex : input.node_vertices) {
        is_node[vertex] = true;
    }
    for (std::size_t index = 0; index < input.poses.size(); ++index) {
        if (!is_node[index]) {
            add_pose(index);
        }
    }
    for (std::size_t index = 0; index < input.node_vertices.size(); ++index) {
        add_pose(input.node_vertices[index]);
        problem.AddParameterBlock(input.velocities[index].data(), 3);
    }
    problem.SetParameterBlockConstant(input.poses[input.fixed_vertex].translation.data());

    // The generic installed correction state also exposes the disabled IMU
    // translation and both disabled cross-axis matrices as constant Ceres
    // blocks.  Retain them in the observed order even though this frozen
    // configuration does not optimize those terms.
    std::array<double, 3> imu_translation = {0.0, 0.0, 0.0};
    problem.AddParameterBlock(&input.calibration.gravity, 1);
    problem.SetParameterLowerBound(&input.calibration.gravity, 0, 0.0);
    problem.AddParameterBlock(imu_translation.data(), 3);
    problem.SetParameterBlockConstant(imu_translation.data());
#if NAVVIS_RECON_CERES_USES_MANIFOLD
    problem.AddParameterBlock(input.calibration.imu_rotation_xyzw.data(), 4,
                              new ceres::EigenQuaternionManifold());
#else
    problem.AddParameterBlock(
        input.calibration.imu_rotation_xyzw.data(), 4,
        new ceres::EigenQuaternionParameterization());
#endif
    problem.AddParameterBlock(input.calibration.linear_acceleration_bias.data(), 3);
    problem.AddParameterBlock(input.calibration.linear_acceleration_scaling.data(), 3);
    problem.AddParameterBlock(input.calibration.linear_acceleration_cross_axis.data(), 6);
    problem.SetParameterBlockConstant(
        input.calibration.linear_acceleration_cross_axis.data());
    problem.AddParameterBlock(input.calibration.angular_velocity_bias.data(), 3);
    problem.AddParameterBlock(input.calibration.angular_velocity_scaling.data(), 3);
    problem.AddParameterBlock(input.calibration.angular_velocity_cross_axis.data(), 6);
    problem.SetParameterBlockConstant(
        input.calibration.angular_velocity_cross_axis.data());

    for (const Edge& edge : input.edges) {
        Pose& source = input.poses[edge.source];
        Pose& target = input.poses[edge.target];
        auto* cost = new ceres::AutoDiffCostFunction<GraphResidual, 6, 4, 3, 4, 3>(
            new GraphResidual(edge));
        ceres::LossFunction* loss =
            edge.kind == 1 ? new ceres::HuberLoss(kLoopClosureHuberScale) : nullptr;
        problem.AddResidualBlock(cost, loss, source.rotation_xyzw.data(),
                                 source.translation.data(), target.rotation_xyzw.data(),
                                 target.translation.data());
    }

    // The installed builder inserts active intrinsic priors before the IMU
    // factors.  A zero three-axis weight suppresses the entire prior block;
    // this is why the default gyro-bias prior is absent from the Ceres graph.
    const std::array<double, 3> zero = {0.0, 0.0, 0.0};
    const std::array<double, 3> one = {1.0, 1.0, 1.0};
    const auto add_vector_prior_if_active = [&problem](
                                                double* parameter,
                                                const std::array<double, 3>& target,
                                                const std::array<double, 3>& weight) {
        if (weight[0] == 0.0 && weight[1] == 0.0 && weight[2] == 0.0) {
            return;
        }
        problem.AddResidualBlock(
            new ceres::AutoDiffCostFunction<VectorPriorResidual, 3, 3>(
                new VectorPriorResidual(target, weight)),
            nullptr, parameter);
    };
    add_vector_prior_if_active(
        input.calibration.linear_acceleration_bias.data(), zero,
        input.calibration_options.linear_acceleration_bias_weight);
    add_vector_prior_if_active(
        input.calibration.linear_acceleration_scaling.data(), one,
        input.calibration_options.linear_acceleration_scaling_weight);
    add_vector_prior_if_active(
        input.calibration.angular_velocity_bias.data(), zero,
        input.calibration_options.angular_velocity_bias_weight);
    add_vector_prior_if_active(
        input.calibration.angular_velocity_scaling.data(), one,
        input.calibration_options.angular_velocity_scaling_weight);

    for (const ImuFactor& factor : input.imu_factors) {
        Pose& source = input.poses[factor.source_vertex];
        Pose& target = input.poses[factor.target_vertex];
        auto* cost = new ceres::AutoDiffCostFunction<
            ImuResidual, 9, 4, 3, 4, 3, 4, 3, 3, 3, 3, 3, 6, 3, 3, 6, 1>(
            new ImuResidual(factor, input.samples, input.imu_rotation_weight,
                            input.imu_velocity_weight, input.imu_position_weight));
        problem.AddResidualBlock(
            cost, nullptr, input.calibration.imu_rotation_xyzw.data(),
            imu_translation.data(), source.rotation_xyzw.data(),
            source.translation.data(), target.rotation_xyzw.data(),
            target.translation.data(), input.velocities[factor.source_velocity].data(),
            input.velocities[factor.target_velocity].data(),
            input.calibration.angular_velocity_bias.data(),
            input.calibration.angular_velocity_scaling.data(),
            input.calibration.angular_velocity_cross_axis.data(),
            input.calibration.linear_acceleration_bias.data(),
            input.calibration.linear_acceleration_scaling.data(),
            input.calibration.linear_acceleration_cross_axis.data(),
            &input.calibration.gravity);
    }

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ImuOrientationPriorResidual, 3, 4>(
            new ImuOrientationPriorResidual()),
        new ceres::ScaledLoss(
            nullptr,
            input.calibration_options.imu_orientation_weight *
                input.calibration_options.imu_orientation_weight,
            ceres::TAKE_OWNERSHIP),
        input.calibration.imu_rotation_xyzw.data());
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<GravityPriorResidual, 1, 1>(
            new GravityPriorResidual(input.calibration_options)),
        nullptr, &input.calibration.gravity);

    ceres::Solver::Options options;
    options.max_num_iterations = static_cast<int>(input.max_iterations);
    options.num_threads = static_cast<int>(input.num_threads);
    ceres::Solver::Summary summary;
    ceres::Solve(options, &problem, &summary);
    if (summary.iterations.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("Ceres iteration count exceeds uint32");
    }
    return SolveResult{summary.initial_cost, summary.final_cost,
                       static_cast<std::uint32_t>(summary.iterations.size()),
                       summary.IsSolutionUsable()};
}

void writeOutput(const std::string& path, const InputProblem& input,
                 const SolveResult& result) {
    LittleEndianWriter writer;
    writer.writeMagic(kOutputMagic);
    writer.writeUint32(kSchemaVersion);
    writer.writeUint32(static_cast<std::uint32_t>(input.poses.size()));
    writer.writeUint32(static_cast<std::uint32_t>(input.velocities.size()));
    writer.writeUint32(result.iterations);
    writer.writeUint32(result.success ? 1U : 0U);
    writer.writeDouble(result.initial_cost);
    writer.writeDouble(result.final_cost);
    writer.writeDouble(input.calibration.gravity);
    for (const double value : input.calibration.imu_rotation_xyzw) {
        writer.writeDouble(value);
    }
    for (const double value : input.calibration.linear_acceleration_bias) {
        writer.writeDouble(value);
    }
    for (const double value : input.calibration.linear_acceleration_scaling) {
        writer.writeDouble(value);
    }
    for (const double value : input.calibration.angular_velocity_bias) {
        writer.writeDouble(value);
    }
    for (const double value : input.calibration.angular_velocity_scaling) {
        writer.writeDouble(value);
    }
    for (const Pose& pose : input.poses) {
        for (const double value : pose.translation) {
            writer.writeDouble(value);
        }
        for (const double value : pose.rotation_xyzw) {
            writer.writeDouble(value);
        }
    }
    for (const auto& velocity : input.velocities) {
        for (const double value : velocity) {
            writer.writeDouble(value);
        }
    }
    writer.save(path);
}

}  // namespace

int main(const int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: " << argv[0] << " INPUT.bin OUTPUT.bin\n";
        return 2;
    }
    try {
        InputProblem input = readInput(argv[1]);
        const SolveResult result = solve(input);
        writeOutput(argv[2], input, result);
        std::cerr << "Ceres Stage2: poses=" << input.poses.size()
                  << " edges=" << input.edges.size()
                  << " imu=" << input.imu_factors.size()
                  << " samples=" << input.samples.size()
                  << " iterations=" << result.iterations
                  << " initial_cost=" << result.initial_cost
                  << " final_cost=" << result.final_cost
                  << " success=" << (result.success ? 1 : 0) << '\n';
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "stage2_imu_ceres_solver: " << error.what() << '\n';
        return 1;
    }
}
