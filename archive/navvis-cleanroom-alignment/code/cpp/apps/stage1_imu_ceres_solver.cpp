// Standalone Ceres worker for the clean-room SurveyorSLAM Stage1 problem.
//
// Command line:
//   navvis_recon_stage1_imu_ceres_solver INPUT.bin OUTPUT.bin
//
// Both files are packed little-endian streams. There is no implicit padding,
// alignment, native-size integer, or trailing data. Quaternions are always
// stored as (x, y, z, w). The input schema is:
//
//   char     magic[8] = "NVSG1CR1"
//   uint32   version = 1
//   uint32   pose_count
//   uint32   node_count
//   uint32   edge_count
//   uint32   accel_count
//   uint32   rotation_count
//   uint32   fixed_vertex
//   uint32   max_iterations
//   uint32   num_threads
//   double   gravity
//   double   imu_q_xyzw[4]
//   Pose     poses[pose_count]
//   Edge     edges[edge_count]
//   Accel    accel_factors[accel_count]
//   Rotation rotation_factors[rotation_count]
//
//   Pose     = double p[3], q_xyzw[4]
//   Edge     = uint32 source, target, kind;  // 0 = odometry, 1 = loop
//              double measured_t[3], measured_q_xyzw[4],
//                     translation_weight, rotation_weight
//   Accel    = uint32 first, second, third;
//              double measurement[3], dt_before, dt_after, loss_dt
//   Rotation = uint32 first, second;
//              double measurement_q_xyzw[4], dt
//
// accel_count must equal node_count - 2 and rotation_count must equal
// node_count - 1. Factor indices are pose-vertex indices, not dense node IDs.
// Loop edges alone use HuberLoss(2500); odometry edges have no robust loss.
//
// The output schema is:
//
//   char   magic[8] = "NVSG1RS1"
//   uint32 version = 1, pose_count, iterations, success
//   double initial_cost, final_cost, gravity, imu_q_xyzw[4]
//   Pose   poses[pose_count]

// success is 1 iff Ceres reports a usable solution. "iterations" is the
// number of minimizer iterations recorded by Ceres.

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
#include <ceres/rotation.h>

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

constexpr std::array<char, 8> kInputMagic = {'N', 'V', 'S', 'G', '1', 'C', 'R', '1'};
constexpr std::array<char, 8> kOutputMagic = {'N', 'V', 'S', 'G', '1', 'R', 'S', '1'};
constexpr std::uint32_t kSchemaVersion = 1;
constexpr double kAccelerationWeight = 50.0;
constexpr double kRotationWeight = 10.0;
constexpr double kLoopHuberScale = 5.0 / 0.002;
constexpr double kGravityPriorTarget = 9.807232;
constexpr double kGravityPriorWeight = 1.0e4;
constexpr double kImuOrientationPriorWeight = 5.0e4;

constexpr std::size_t kInputHeaderBytes = 8 + 9 * 4 + 5 * 8;
constexpr std::size_t kPoseBytes = 7 * 8;
constexpr std::size_t kEdgeBytes = 3 * 4 + 9 * 8;
constexpr std::size_t kAccelerationBytes = 3 * 4 + 6 * 8;
constexpr std::size_t kRotationBytes = 2 * 4 + 5 * 8;

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

struct AccelerationFactor {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    std::uint32_t third = 0;
    std::array<double, 3> measurement{};
    double dt_before = 0.0;
    double dt_after = 0.0;
    double loss_dt = 0.0;
};

struct RotationFactor {
    std::uint32_t first = 0;
    std::uint32_t second = 0;
    std::array<double, 4> measurement_xyzw{};
    double dt = 0.0;
};

struct InputProblem {
    std::uint32_t node_count = 0;
    std::uint32_t fixed_vertex = 0;
    std::uint32_t max_iterations = 0;
    std::uint32_t num_threads = 0;
    double gravity = 0.0;
    std::array<double, 4> imu_rotation_xyzw{};
    std::vector<Pose> poses;
    std::vector<Edge> edges;
    std::vector<AccelerationFactor> acceleration_factors;
    std::vector<RotationFactor> rotation_factors;
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

std::size_t checkedAddRecords(const std::size_t base, const std::uint32_t count,
                              const std::size_t record_size) {
    const std::size_t count_size = static_cast<std::size_t>(count);
    if (count_size > (std::numeric_limits<std::size_t>::max() - base) / record_size) {
        throw std::runtime_error("input record counts overflow the address space");
    }
    return base + count_size * record_size;
}

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

void requireNonzeroQuaternion(
    const std::array<double, 4>& quaternion, const std::string& name) {
    requireFinite(quaternion, name);
    double squared_norm = 0.0;
    for (const double value : quaternion) {
        squared_norm += value * value;
    }
    if (!(squared_norm > std::numeric_limits<double>::min())) {
        throw std::runtime_error(name + " has zero norm");
    }
}

void normalizeQuaternion(std::array<double, 4>& quaternion, const std::string& name) {
    requireNonzeroQuaternion(quaternion, name);
    double squared_norm = 0.0;
    for (const double value : quaternion) {
        squared_norm += value * value;
    }
    const double inverse_norm = 1.0 / std::sqrt(squared_norm);
    for (double& value : quaternion) {
        value *= inverse_norm;
    }
}

void requirePositive(const double value, const std::string& name) {
    requireFinite(value, name);
    if (!(value > 0.0)) {
        throw std::runtime_error(name + " must be positive");
    }
}

InputProblem readInput(const std::string& path) {
    LittleEndianReader reader(path);
    if (reader.readMagic() != kInputMagic) {
        throw std::runtime_error("input magic is not NVSG1CR1");
    }
    const std::uint32_t version = reader.readUint32();
    if (version != kSchemaVersion) {
        throw std::runtime_error("unsupported input schema version: " +
                                 std::to_string(version));
    }

    const std::uint32_t pose_count = reader.readUint32();
    const std::uint32_t node_count = reader.readUint32();
    const std::uint32_t edge_count = reader.readUint32();
    const std::uint32_t acceleration_count = reader.readUint32();
    const std::uint32_t rotation_count = reader.readUint32();

    InputProblem input;
    input.node_count = node_count;
    input.fixed_vertex = reader.readUint32();
    input.max_iterations = reader.readUint32();
    input.num_threads = reader.readUint32();
    input.gravity = reader.readDouble();
    input.imu_rotation_xyzw = readDoubleArray<4>(reader);

    std::size_t expected_size = kInputHeaderBytes;
    expected_size = checkedAddRecords(expected_size, pose_count, kPoseBytes);
    expected_size = checkedAddRecords(expected_size, edge_count, kEdgeBytes);
    expected_size =
        checkedAddRecords(expected_size, acceleration_count, kAccelerationBytes);
    expected_size = checkedAddRecords(expected_size, rotation_count, kRotationBytes);
    if (reader.size() != expected_size) {
        throw std::runtime_error("input size does not match header record counts");
    }
    if (pose_count == 0 || node_count < 2 || node_count > pose_count) {
        throw std::runtime_error("pose_count/node_count are inconsistent");
    }
    if (acceleration_count != node_count - 2 || rotation_count != node_count - 1) {
        throw std::runtime_error("Stage1 factor counts do not match node_count");
    }
    if (input.fixed_vertex >= pose_count) {
        throw std::runtime_error("fixed_vertex is outside the pose array");
    }
    if (input.max_iterations == 0 ||
        input.max_iterations > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("max_iterations is outside the supported range");
    }
    if (input.num_threads == 0 ||
        input.num_threads > static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error("num_threads is outside the supported range");
    }
    requirePositive(input.gravity, "gravity");
    normalizeQuaternion(input.imu_rotation_xyzw, "imu_q_xyzw");

    input.poses.resize(pose_count);
    for (std::uint32_t index = 0; index < pose_count; ++index) {
        Pose& pose = input.poses[index];
        pose.translation = readDoubleArray<3>(reader);
        pose.rotation_xyzw = readDoubleArray<4>(reader);
        requireFinite(pose.translation, "pose translation");
        // Online Surveyor passes frontend and warm-start coefficients straight
        // to Ceres. In particular, the first raw-IMU pose is intentionally a
        // few ulps away from unit length before the first update.
        requireNonzeroQuaternion(pose.rotation_xyzw, "pose quaternion");
    }

    input.edges.resize(edge_count);
    for (std::uint32_t index = 0; index < edge_count; ++index) {
        Edge& edge = input.edges[index];
        edge.source = reader.readUint32();
        edge.target = reader.readUint32();
        edge.kind = reader.readUint32();
        edge.measured_translation = readDoubleArray<3>(reader);
        edge.measured_rotation_xyzw = readDoubleArray<4>(reader);
        edge.translation_weight = reader.readDouble();
        edge.rotation_weight = reader.readDouble();
        if (edge.source >= pose_count || edge.target >= pose_count || edge.kind > 1) {
            throw std::runtime_error("edge has an invalid vertex or kind");
        }
        requireFinite(edge.measured_translation, "edge translation");
        // Surveyor's graph constraints preserve the frontend's original
        // near-unit quaternion coefficients and sign.  Ceres evaluates those
        // coefficients directly in the residual; normalizing here perturbs
        // the installed problem by several ulps.
        requireNonzeroQuaternion(edge.measured_rotation_xyzw, "edge quaternion");
        requirePositive(edge.translation_weight, "edge translation weight");
        requirePositive(edge.rotation_weight, "edge rotation weight");
    }

    input.acceleration_factors.resize(acceleration_count);
    for (std::uint32_t index = 0; index < acceleration_count; ++index) {
        AccelerationFactor& factor = input.acceleration_factors[index];
        factor.first = reader.readUint32();
        factor.second = reader.readUint32();
        factor.third = reader.readUint32();
        factor.measurement = readDoubleArray<3>(reader);
        factor.dt_before = reader.readDouble();
        factor.dt_after = reader.readDouble();
        factor.loss_dt = reader.readDouble();
        if (factor.first >= pose_count || factor.second >= pose_count ||
            factor.third >= pose_count) {
            throw std::runtime_error("acceleration factor has an invalid vertex");
        }
        requireFinite(factor.measurement, "acceleration measurement");
        requirePositive(factor.dt_before, "acceleration dt_before");
        requirePositive(factor.dt_after, "acceleration dt_after");
        requirePositive(factor.loss_dt, "acceleration loss_dt");
    }

    input.rotation_factors.resize(rotation_count);
    for (std::uint32_t index = 0; index < rotation_count; ++index) {
        RotationFactor& factor = input.rotation_factors[index];
        factor.first = reader.readUint32();
        factor.second = reader.readUint32();
        factor.measurement_xyzw = readDoubleArray<4>(reader);
        factor.dt = reader.readDouble();
        if (factor.first >= pose_count || factor.second >= pose_count) {
            throw std::runtime_error("rotation factor has an invalid vertex");
        }
        requireNonzeroQuaternion(
            factor.measurement_xyzw, "rotation measurement");
        requirePositive(factor.dt, "rotation dt");
    }
    if (reader.offset() != reader.size()) {
        throw std::runtime_error("input contains trailing bytes");
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
void writeQuaternionVectorResidual(const Eigen::Quaternion<T>& quaternion,
                                   const double scale, T* residual) {
    const T sign = quaternion.w() < T(0.0) ? T(-1.0) : T(1.0);
    residual[0] = T(2.0 * scale) * sign * quaternion.x();
    residual[1] = T(2.0 * scale) * sign * quaternion.y();
    residual[2] = T(2.0 * scale) * sign * quaternion.z();
}

// Cartographer's ConstantYawQuaternionPlus, adapted to Eigen's xyzw storage.
// The increment is right-multiplied so its xy-plane is the submap frame.
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
        const Eigen::Quaternion<T> rotation_error =
            quaternionFromXyzw<T>(measured_rotation_xyzw).conjugate() *
            predicted_rotation;
        writeQuaternionVectorResidual(rotation_error, rotation_weight, residual + 3);
        return true;
    }

    std::array<double, 3> measured_translation;
    std::array<double, 4> measured_rotation_xyzw;
    double translation_weight;
    double rotation_weight;
};

struct AccelerationResidual {
    explicit AccelerationResidual(const AccelerationFactor& factor)
        : measurement(factor.measurement),
          dt_before(factor.dt_before),
          dt_after(factor.dt_after),
          scale(kAccelerationWeight / factor.loss_dt) {}

    template <typename T>
    bool operator()(const T* middle_rotation, const T* start_position,
                    const T* middle_position, const T* end_position,
                    const T* gravity, const T* imu_rotation, T* residual) const {
        const Eigen::Quaternion<T> middle = quaternionFromXyzw(middle_rotation);
        const Eigen::Quaternion<T> imu = quaternionFromXyzw(imu_rotation);
        const Eigen::Matrix<T, 3, 1> measured{
            T(measurement[0]), T(measurement[1]), T(measurement[2])};
        Eigen::Matrix<T, 3, 1> measured_world_delta_velocity = middle * (imu * measured);
        measured_world_delta_velocity[2] -=
            gravity[0] * T(0.5 * (dt_before + dt_after));

        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> start(start_position);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> center(middle_position);
        const Eigen::Map<const Eigen::Matrix<T, 3, 1>> end(end_position);
        const Eigen::Matrix<T, 3, 1> delta_velocity =
            (end - center) / T(dt_after) - (center - start) / T(dt_before);
        Eigen::Map<Eigen::Matrix<T, 3, 1>> residual_map(residual);
        residual_map = T(scale) * (measured_world_delta_velocity - delta_velocity);
        return true;
    }

    std::array<double, 3> measurement;
    double dt_before;
    double dt_after;
    double scale;
};

struct RotationResidual {
    explicit RotationResidual(const RotationFactor& factor)
        : measurement_xyzw(factor.measurement_xyzw),
          scale(kRotationWeight / factor.dt) {}

    template <typename T>
    bool operator()(const T* first_rotation, const T* second_rotation,
                    const T* imu_rotation, T* residual) const {
        const Eigen::Quaternion<T> first = quaternionFromXyzw(first_rotation);
        const Eigen::Quaternion<T> second = quaternionFromXyzw(second_rotation);
        const Eigen::Quaternion<T> imu = quaternionFromXyzw(imu_rotation);
        const Eigen::Quaternion<T> measurement =
            quaternionFromXyzw<T>(measurement_xyzw);
        const Eigen::Quaternion<T> error =
            imu * measurement.conjugate() * imu.conjugate() * first.conjugate() * second;
        writeQuaternionVectorResidual(error, scale, residual);
        return true;
    }

    std::array<double, 4> measurement_xyzw;
    double scale;
};

struct GravityPriorResidual {
    template <typename T>
    bool operator()(const T* gravity, T* residual) const {
        residual[0] = T(kGravityPriorWeight) * (gravity[0] - T(kGravityPriorTarget));
        return true;
    }
};

struct ImuOrientationPriorResidual {
    template <typename T>
    bool operator()(const T* imu_rotation, T* residual) const {
        const T quaternion_wxyz[4] = {imu_rotation[3], imu_rotation[0],
                                      imu_rotation[1], imu_rotation[2]};
        T angle_axis[3];
        ceres::QuaternionToAngleAxis(quaternion_wxyz, angle_axis);
        residual[0] = T(kImuOrientationPriorWeight) * angle_axis[0];
        residual[1] = T(kImuOrientationPriorWeight) * angle_axis[1];
        residual[2] = T(kImuOrientationPriorWeight) * angle_axis[2];
        return true;
    }
};

struct SolveResult {
    double initial_cost = 0.0;
    double final_cost = 0.0;
    std::uint32_t iterations = 0;
    bool success = false;
};

SolveResult solve(InputProblem& input) {
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
    // SparsePoseGraph owns submaps separately from trajectory nodes and adds
    // their parameter blocks first.  Ceres preserves this insertion order in
    // the reduced program; changing it changes sparse elimination low bits.
    for (std::size_t index = input.node_count; index < input.poses.size(); ++index) {
        add_pose(index);
    }
    for (std::size_t index = 0; index < input.node_count; ++index) {
        add_pose(index);
    }
    problem.SetParameterBlockConstant(input.poses[input.fixed_vertex].translation.data());

    // The installed IMU calibration is a full Rigid3.  Its translation is an
    // unused constant block in Stage1, but retaining it reproduces the exact
    // parameter-program layout before the live rotation and gravity blocks.
    std::array<double, 3> imu_translation{};
    problem.AddParameterBlock(imu_translation.data(), 3);
    problem.SetParameterBlockConstant(imu_translation.data());
#if NAVVIS_RECON_CERES_USES_MANIFOLD
    problem.AddParameterBlock(input.imu_rotation_xyzw.data(), 4,
                              new ceres::EigenQuaternionManifold());
#else
    problem.AddParameterBlock(input.imu_rotation_xyzw.data(), 4,
                              new ceres::EigenQuaternionParameterization());
#endif
    problem.AddParameterBlock(&input.gravity, 1);
    problem.SetParameterLowerBound(&input.gravity, 0, 0.0);

    for (const Edge& edge : input.edges) {
        Pose& source = input.poses[edge.source];
        Pose& target = input.poses[edge.target];
        auto* cost = new ceres::AutoDiffCostFunction<GraphResidual, 6, 4, 3, 4, 3>(
            new GraphResidual(edge));
        ceres::LossFunction* loss =
            edge.kind == 1 ? static_cast<ceres::LossFunction*>(new ceres::HuberLoss(kLoopHuberScale))
                           : nullptr;
        problem.AddResidualBlock(cost, loss, source.rotation_xyzw.data(),
                                 source.translation.data(), target.rotation_xyzw.data(),
                                 target.translation.data());
    }

    for (const AccelerationFactor& factor : input.acceleration_factors) {
        Pose& first = input.poses[factor.first];
        Pose& second = input.poses[factor.second];
        Pose& third = input.poses[factor.third];
        auto* cost = new ceres::AutoDiffCostFunction<AccelerationResidual, 3, 4, 3, 3, 3,
                                                     1, 4>(
            new AccelerationResidual(factor));
        problem.AddResidualBlock(cost, nullptr, second.rotation_xyzw.data(),
                                 first.translation.data(), second.translation.data(),
                                 third.translation.data(), &input.gravity,
                                 input.imu_rotation_xyzw.data());
    }

    for (const RotationFactor& factor : input.rotation_factors) {
        Pose& first = input.poses[factor.first];
        Pose& second = input.poses[factor.second];
        auto* cost = new ceres::AutoDiffCostFunction<RotationResidual, 3, 4, 4, 4>(
            new RotationResidual(factor));
        problem.AddResidualBlock(cost, nullptr, first.rotation_xyzw.data(),
                                 second.rotation_xyzw.data(), input.imu_rotation_xyzw.data());
    }

    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<GravityPriorResidual, 1, 1>(
            new GravityPriorResidual()),
        nullptr, &input.gravity);
    problem.AddResidualBlock(
        new ceres::AutoDiffCostFunction<ImuOrientationPriorResidual, 3, 4>(
            new ImuOrientationPriorResidual()),
        nullptr, input.imu_rotation_xyzw.data());

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
    writer.writeUint32(result.iterations);
    writer.writeUint32(result.success ? 1U : 0U);
    writer.writeDouble(result.initial_cost);
    writer.writeDouble(result.final_cost);
    writer.writeDouble(input.gravity);
    for (const double value : input.imu_rotation_xyzw) {
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
        std::cerr << "Ceres Stage1: poses=" << input.poses.size()
                  << " edges=" << input.edges.size()
                  << " accel=" << input.acceleration_factors.size()
                  << " rotation=" << input.rotation_factors.size()
                  << " iterations=" << result.iterations
                  << " initial_cost=" << result.initial_cost
                  << " final_cost=" << result.final_cost
                  << " success=" << (result.success ? 1 : 0) << '\n';
        return result.success ? 0 : 1;
    } catch (const std::exception& error) {
        std::cerr << "stage1_imu_ceres_solver: " << error.what() << '\n';
        return 1;
    }
}
