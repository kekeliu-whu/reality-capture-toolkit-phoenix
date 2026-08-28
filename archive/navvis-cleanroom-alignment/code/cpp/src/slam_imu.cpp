#include "navvis_recon/slam_imu.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace navvis_recon::slam {
namespace {

bool finiteVector(const Eigen::Vector3d& value) {
    return value.array().isFinite().all();
}

bool finiteQuaternion(const Eigen::Quaterniond& value) {
    return value.coeffs().array().isFinite().all();
}

double binaryQuaternionSquaredNorm(const Eigen::Quaterniond& quaternion) {
    const double x = quaternion.x();
    const double y = quaternion.y();
    const double z = quaternion.z();
    const double w = quaternion.w();
    return (z * z + x * x) + (w * w + y * y);
}

Eigen::Quaterniond binaryQuaternionInverse(const Eigen::Quaterniond& quaternion) {
    const double squared_norm = binaryQuaternionSquaredNorm(quaternion);
    if (!(squared_norm > 0.0)) {
        throw std::invalid_argument("quaternion must be non-zero");
    }
    return Eigen::Quaterniond(
        quaternion.w() / squared_norm,
        -quaternion.x() / squared_norm,
        -quaternion.y() / squared_norm,
        -quaternion.z() / squared_norm);
}

Eigen::Quaterniond binaryQuaternionProduct(
    const Eigen::Quaterniond& lhs,
    const Eigen::Quaterniond& rhs) {
    const double ax = lhs.x();
    const double ay = lhs.y();
    const double az = lhs.z();
    const double aw = lhs.w();
    const double bx = rhs.x();
    const double by = rhs.y();
    const double bz = rhs.z();
    const double bw = rhs.w();

    // Grouped like the packed-double Eigen kernel recovered from the frontend binary.
    const double xy_t1_0 = aw * bx + ay * bz;
    const double xy_t1_1 = aw * by + ay * bw;
    const double xy_t2_0 = az * bx - ax * bz;
    const double xy_t2_1 = az * by - ax * bw;
    const double x = xy_t1_0 - xy_t2_1;
    const double y = xy_t1_1 + xy_t2_0;

    const double zw_t1_0 = aw * bz - ay * bx;
    const double zw_t1_1 = aw * bw - ay * by;
    const double zw_t2_0 = az * bz + ax * bx;
    const double zw_t2_1 = az * bw + ax * by;
    const double z = zw_t1_0 + zw_t2_1;
    const double w = zw_t1_1 - zw_t2_0;
    return Eigen::Quaterniond(w, x, y, z);
}

Eigen::Quaterniond arrayQuaternionProduct(
    const Eigen::Quaterniond& lhs,
    const Eigen::Quaterniond& rhs) {
    const double ax = lhs.x();
    const double ay = lhs.y();
    const double az = lhs.z();
    const double aw = lhs.w();
    const double bx = rhs.x();
    const double by = rhs.y();
    const double bz = rhs.z();
    const double bw = rhs.w();
    return Eigen::Quaterniond(
        aw * bw - ((ax * bx + ay * by) + az * bz),
        (aw * bx + bw * ax) + (ay * bz - az * by),
        (aw * by + bw * ay) + (az * bx - ax * bz),
        (aw * bz + bw * az) + (ax * by - ay * bx));
}

Eigen::Quaterniond binaryQuaternionNormalized(const Eigen::Quaterniond& quaternion) {
    const double norm = std::sqrt(binaryQuaternionSquaredNorm(quaternion));
    if (!(norm > 0.0)) {
        throw std::invalid_argument("quaternion must be non-zero");
    }
    return Eigen::Quaterniond(
        quaternion.w() / norm,
        quaternion.x() / norm,
        quaternion.y() / norm,
        quaternion.z() / norm);
}

Eigen::Vector3d binaryQuaternionRotate(
    const Eigen::Quaterniond& quaternion,
    const Eigen::Vector3d& vector) {
    const double qx = quaternion.x();
    const double qy = quaternion.y();
    const double qz = quaternion.z();
    const double qw = quaternion.w();
    const double vx = vector.x();
    const double vy = vector.y();
    const double vz = vector.z();
    const double tx = 2.0 * (qy * vz - qz * vy);
    const double ty = 2.0 * (qz * vx - qx * vz);
    const double tz = 2.0 * (qx * vy - qy * vx);
    const double cx = qy * tz - qz * ty;
    const double cy = qz * tx - qx * tz;
    const double cz = qx * ty - qy * tx;
    return Eigen::Vector3d(
        (vx + qw * tx) + cx,
        (vy + qw * ty) + cy,
        (vz + qw * tz) + cz);
}

Eigen::Vector3d rawQuaternionTransformVector(
    const Eigen::Quaterniond& quaternion,
    const Eigen::Vector3d& vector) {
    const double qx = quaternion.x();
    const double qy = quaternion.y();
    const double qz = quaternion.z();
    const double qw = quaternion.w();
    const double vx = vector.x();
    const double vy = vector.y();
    const double vz = vector.z();
    double tx = qy * vz - qz * vy;
    tx = tx + tx;
    double ty = qz * vx - qx * vz;
    ty = ty + ty;
    double tz = qx * vy - qy * vx;
    tz = tz + tz;
    return Eigen::Vector3d(
        (qy * tz - qz * ty) + (qw * tx + vx),
        (qz * tx - qx * tz) + (qw * ty + vy),
        (qx * ty - qy * tx) + (vz + qw * tz));
}

Eigen::Quaterniond rotationVectorQuaternion(const Eigen::Vector3d& rotation_vector) {
    const double x = rotation_vector.x();
    const double y = rotation_vector.y();
    const double z = rotation_vector.z();
    const double squared_angle = z * z + (x * x + y * y);
    if (!(squared_angle > 0.0)) {
        return Eigen::Quaterniond::Identity();
    }
    const double angle = std::sqrt(squared_angle);
    const double half_angle = 0.5 * angle;
    const double coefficient = std::sin(half_angle) / angle;
    return Eigen::Quaterniond(
        std::cos(half_angle), coefficient * x, coefficient * y, coefficient * z);
}

Eigen::Quaterniond quaternionFromTwoVectors(
    const Eigen::Vector3d& first,
    const Eigen::Vector3d& second) {
    const double first_norm = std::sqrt(
        (first.x() * first.x() + first.y() * first.y()) + first.z() * first.z());
    const double second_norm = std::sqrt(
        (second.x() * second.x() + second.y() * second.y()) + second.z() * second.z());
    if (!(first_norm > 0.0) || !(second_norm > 0.0)) {
        return Eigen::Quaterniond::Identity();
    }

    const double first_x = first.x() / first_norm;
    const double first_y = first.y() / first_norm;
    const double first_z = first.z() / first_norm;
    const double second_x = second.x() / second_norm;
    const double second_y = second.y() / second_norm;
    const double second_z = second.z() / second_norm;
    const double dot =
        (first_x * second_x + first_y * second_y) + first_z * second_z;
    if (dot < -1.0 + std::numeric_limits<double>::epsilon()) {
        return Eigen::Quaterniond::FromTwoVectors(first, second);
    }

    const double doubled_one_plus_dot = (dot + 1.0) + (dot + 1.0);
    const double scale = std::sqrt(doubled_one_plus_dot);
    const double inverse_scale = 1.0 / scale;
    return Eigen::Quaterniond(
        scale * 0.5,
        (first_y * second_z - first_z * second_y) * inverse_scale,
        (first_z * second_x - first_x * second_z) * inverse_scale,
        (first_x * second_y - first_y * second_x) * inverse_scale);
}

Eigen::Vector3d initializeGravity(
    const Eigen::Quaterniond& orientation,
    double gravity_magnitude) {
    const Eigen::Quaterniond inverse = binaryQuaternionInverse(orientation);
    const double x = inverse.x();
    const double y = inverse.y();
    const double z = inverse.z();
    const double w = inverse.w();
    const double two_y = y + y;
    const double two_negative_x = (-x) + (-x);
    return Eigen::Vector3d(
        ((0.0 - z * two_negative_x) + w * two_y) * gravity_magnitude,
        ((two_y * z - x * 0.0) + w * two_negative_x) * gravity_magnitude,
        ((two_negative_x * x - two_y * y) + 1.0) * gravity_magnitude);
}

void angularUpdate(
    double dt_s,
    const Eigen::Quaterniond& orientation,
    const Eigen::Vector3d& gravity,
    const Eigen::Vector3d& angular_start,
    const Eigen::Vector3d& angular_end,
    Eigen::Quaterniond* output_orientation,
    Eigen::Vector3d* output_gravity) {
    const Eigen::Vector3d angular_sum(
        angular_start.x() + angular_end.x(),
        angular_start.y() + angular_end.y(),
        angular_start.z() + angular_end.z());
    const Eigen::Vector3d rotation_vector = (0.5 * angular_sum) * dt_s;
    const Eigen::Quaterniond delta = rotationVectorQuaternion(rotation_vector);
    *output_orientation = binaryQuaternionProduct(orientation, delta);
    *output_gravity = binaryQuaternionInverse(delta) * gravity;
}

void gravityUpdate(
    double alpha,
    const Eigen::Quaterniond& input_orientation,
    const Eigen::Vector3d& previous_gravity,
    const Eigen::Vector3d& measured_acceleration,
    Eigen::Quaterniond* output_orientation,
    Eigen::Vector3d* output_gravity) {
    const Eigen::Vector3d world_gravity = input_orientation * previous_gravity;
    const Eigen::Vector3d world_acceleration = input_orientation * measured_acceleration;
    const Eigen::Vector3d blended_world_gravity =
        (1.0 - alpha) * world_gravity + alpha * world_acceleration;
    const Eigen::Quaterniond inverse_orientation = input_orientation.inverse();
    *output_gravity = inverse_orientation * blended_world_gravity;
    const Eigen::Vector3d expected_gravity = inverse_orientation * Eigen::Vector3d::UnitZ();
    const Eigen::Quaterniond correction =
        quaternionFromTwoVectors(*output_gravity, expected_gravity);
    *output_orientation = binaryQuaternionNormalized(
        binaryQuaternionProduct(input_orientation, correction));
}

bool equalCoefficients(
    const Eigen::Quaterniond& first,
    const Eigen::Quaterniond& second) noexcept {
    return first.x() == second.x() && first.y() == second.y() &&
           first.z() == second.z() && first.w() == second.w();
}

Eigen::Vector3d predictTranslation(
    TimestampNs previous_timestamp_ns,
    const Eigen::Vector3d& previous_translation,
    TimestampNs anchor_timestamp_ns,
    const Eigen::Vector3d& anchor_translation,
    const Eigen::Quaterniond& anchor_rotation,
    TimestampNs query_timestamp_ns) {
    if (anchor_timestamp_ns <= previous_timestamp_ns ||
        query_timestamp_ns < anchor_timestamp_ns) {
        throw std::invalid_argument("constant-velocity prediction timestamps are invalid");
    }
    const double pose_delta_seconds =
        static_cast<double>(anchor_timestamp_ns - previous_timestamp_ns) / 1.0e9;
    const double inverse_pose_delta_seconds = 1.0 / pose_delta_seconds;
    const Eigen::Vector3d linear_velocity =
        (anchor_translation - previous_translation) * inverse_pose_delta_seconds;
    const double extrapolation_seconds =
        static_cast<double>(query_timestamp_ns - anchor_timestamp_ns) / 1.0e9;
    const Eigen::Vector3d displacement = extrapolation_seconds * linear_velocity;
    const Eigen::Vector3d local_displacement =
        binaryQuaternionRotate(binaryQuaternionInverse(anchor_rotation), displacement);
    return anchor_translation + binaryQuaternionRotate(anchor_rotation, local_displacement);
}

RigidPose endFromStartPoseWithRoundtrip(
    const RigidPose& world_from_start,
    const RigidPose& world_from_end) {
    const Eigen::Quaterniond start_from_world_rotation =
        world_from_start.rotation.inverse();
    const Eigen::Vector3d start_from_world_translation =
        start_from_world_rotation * -world_from_start.translation;
    const Eigen::Quaterniond roundtrip_world_from_start_rotation =
        start_from_world_rotation.inverse();
    const Eigen::Vector3d roundtrip_world_from_start_translation =
        roundtrip_world_from_start_rotation * -start_from_world_translation;

    const Eigen::Quaterniond end_from_world_rotation = world_from_end.rotation.inverse();
    const Eigen::Vector3d end_from_world_translation =
        end_from_world_rotation * -world_from_end.translation;
    return RigidPose{
        end_from_world_translation +
            end_from_world_rotation * roundtrip_world_from_start_translation,
        end_from_world_rotation * roundtrip_world_from_start_rotation};
}

void validatePose(const RigidPose& pose) {
    if (!finiteVector(pose.translation) || !finiteQuaternion(pose.rotation) ||
        !(binaryQuaternionSquaredNorm(pose.rotation) > 0.0)) {
        throw std::invalid_argument("pose must contain finite translation and non-zero rotation");
    }
}

}  // namespace

RigidPose RigidPose::identity() noexcept {
    return RigidPose{};
}

double secondsFromNanoseconds(TimestampNs duration_ns) noexcept {
    return static_cast<double>(duration_ns) / 1.0e9;
}

RawImuTracker::RawImuTracker(
    std::vector<ImuSample> samples,
    RawImuTrackerOptions options)
    : samples_(std::move(samples)), options_(options) {
    if (samples_.size() < 2) {
        throw std::invalid_argument("raw IMU tracking needs at least two samples");
    }
    if (!(options_.initial_gravity_time_constant_s > 0.0) ||
        !(options_.steady_state_gravity_time_constant_s > 0.0) ||
        !(options_.time_constant_fade_duration_s > 0.0) ||
        !(options_.max_gravity_norm_error_mps2 >= 0.0) ||
        !(options_.gravity_mps2 > 0.0)) {
        throw std::invalid_argument("raw IMU tracker options are invalid");
    }
    for (std::size_t index = 0; index < samples_.size(); ++index) {
        const ImuSample& sample = samples_[index];
        if (!finiteVector(sample.linear_acceleration) ||
            !finiteVector(sample.angular_velocity) ||
            !finiteQuaternion(sample.orientation) ||
            !(binaryQuaternionSquaredNorm(sample.orientation) > 0.0)) {
            throw std::invalid_argument("raw IMU sample contains invalid values");
        }
        if (index > 0 && sample.timestamp_ns <= samples_[index - 1].timestamp_ns) {
            throw std::invalid_argument("raw IMU samples must be strictly time ordered");
        }
    }
}

bool RawImuTracker::initialized() const noexcept {
    return time_ns_.has_value();
}

std::optional<TimestampNs> RawImuTracker::timeNs() const noexcept {
    return time_ns_;
}

const Eigen::Quaterniond& RawImuTracker::orientation() const {
    if (!initialized()) {
        throw std::logic_error("the IMU tracker has not been initialized");
    }
    return orientation_;
}

Eigen::Vector3d RawImuTracker::gravityObservation() const {
    if (!initialized()) {
        throw std::logic_error("the IMU tracker has not been initialized");
    }
    return rawQuaternionTransformVector(
        binaryQuaternionInverse(orientation_),
        Eigen::Vector3d(0.0, 0.0, options_.gravity_mps2));
}

RawImuTracker::Bracket RawImuTracker::bracket(TimestampNs timestamp_ns) const {
    const auto high_iterator = std::lower_bound(
        samples_.begin(),
        samples_.end(),
        timestamp_ns,
        [](const ImuSample& sample, TimestampNs timestamp) {
            return sample.timestamp_ns < timestamp;
        });
    const std::size_t high = static_cast<std::size_t>(high_iterator - samples_.begin());
    if (high == 0) {
        return Bracket{0, 1};
    }
    if (high >= samples_.size()) {
        return Bracket{samples_.size() - 2, samples_.size() - 1};
    }
    return Bracket{high - 1, high};
}

Eigen::Quaterniond RawImuTracker::interpolateOrientation(TimestampNs timestamp_ns) const {
    const Bracket interval = bracket(timestamp_ns);
    const Eigen::Quaterniond& first = samples_[interval.low].orientation;
    Eigen::Quaterniond second = samples_[interval.high].orientation;
    if (first.coeffs().dot(second.coeffs()) < 0.0) {
        second.coeffs() *= -1.0;
    }
    const TimestampNs duration =
        samples_[interval.high].timestamp_ns - samples_[interval.low].timestamp_ns;
    const double remaining = static_cast<double>(
        samples_[interval.high].timestamp_ns - timestamp_ns) /
        static_cast<double>(duration);
    const double elapsed = 1.0 - remaining;
    return Eigen::Quaterniond(
        remaining * first.w() + elapsed * second.w(),
        remaining * first.x() + elapsed * second.x(),
        remaining * first.y() + elapsed * second.y(),
        remaining * first.z() + elapsed * second.z());
}

double RawImuTracker::gravityTimeConstant(double elapsed_s) const noexcept {
    const double fade_elapsed = elapsed_s - options_.time_constant_init_duration_s;
    if (fade_elapsed <= 0.0) {
        return options_.initial_gravity_time_constant_s;
    }
    if (fade_elapsed >= options_.time_constant_fade_duration_s) {
        return options_.steady_state_gravity_time_constant_s;
    }
    const double alpha = fade_elapsed / options_.time_constant_fade_duration_s;
    return options_.initial_gravity_time_constant_s +
           alpha * (options_.steady_state_gravity_time_constant_s -
                    options_.initial_gravity_time_constant_s);
}

const Eigen::Quaterniond& RawImuTracker::advance(TimestampNs timestamp_ns) {
    if (timestamp_ns < samples_.front().timestamp_ns ||
        timestamp_ns > samples_.back().timestamp_ns) {
        throw std::out_of_range("IMU prediction timestamp is outside sample support");
    }
    if (!initialized()) {
        if (options_.init_tilt_from_imu_orientation) {
            orientation_ = interpolateOrientation(timestamp_ns);
            gravity_vector_ = initializeGravity(orientation_, options_.gravity_mps2);
        } else {
            orientation_ = Eigen::Quaterniond::Identity();
            gravity_vector_ = Eigen::Vector3d(0.0, 0.0, options_.gravity_mps2);
        }
        time_ns_ = timestamp_ns;
        initial_time_ns_ = timestamp_ns;
        following_index_ = static_cast<std::size_t>(std::upper_bound(
            samples_.begin(),
            samples_.end(),
            timestamp_ns,
            [](TimestampNs timestamp, const ImuSample& sample) {
                return timestamp < sample.timestamp_ns;
            }) - samples_.begin());
        return orientation_;
    }
    if (timestamp_ns < *time_ns_) {
        throw std::invalid_argument("raw IMU tracker cannot advance backwards");
    }

    while (*time_ns_ < timestamp_ns) {
        if (following_index_ >= samples_.size()) {
            throw std::out_of_range("IMU prediction timestamp is outside sample support");
        }
        const std::size_t following = following_index_;
        const TimestampNs interval_end_ns =
            std::min(timestamp_ns, samples_[following].timestamp_ns);
        const TimestampNs low_timestamp_ns = samples_[following - 1].timestamp_ns;
        const TimestampNs high_timestamp_ns = samples_[following].timestamp_ns;
        const double sample_span_s =
            secondsFromNanoseconds(high_timestamp_ns - low_timestamp_ns);
        const double elapsed_in_sample_s =
            secondsFromNanoseconds(*time_ns_ - low_timestamp_ns);
        const double remaining_in_sample_s =
            secondsFromNanoseconds(high_timestamp_ns - interval_end_ns);
        const double angular_dt_s =
            sample_span_s - elapsed_in_sample_s - remaining_in_sample_s;
        const double dt_s = secondsFromNanoseconds(interval_end_ns - *time_ns_);

        const double start_fraction = elapsed_in_sample_s / sample_span_s;
        const Eigen::Vector3d angular_start =
            start_fraction * samples_[following].angular_velocity +
            (1.0 - start_fraction) * samples_[following - 1].angular_velocity;
        const double remaining_fraction = remaining_in_sample_s / sample_span_s;
        const Eigen::Vector3d angular_end =
            remaining_fraction * samples_[following - 1].angular_velocity +
            (1.0 - remaining_fraction) * samples_[following].angular_velocity;
        angularUpdate(
            angular_dt_s,
            orientation_,
            gravity_vector_,
            angular_start,
            angular_end,
            &orientation_,
            &gravity_vector_);

        const double acceleration_fraction =
            (elapsed_in_sample_s + dt_s) / sample_span_s;
        const Eigen::Vector3d acceleration =
            acceleration_fraction * samples_[following].linear_acceleration +
            (1.0 - acceleration_fraction) *
                samples_[following - 1].linear_acceleration;
        const double elapsed_s = secondsFromNanoseconds(timestamp_ns - initial_time_ns_);
        const double time_constant = gravityTimeConstant(elapsed_s);
        const bool acceleration_is_valid =
            std::abs(acceleration.norm() - options_.gravity_mps2) <=
            options_.max_gravity_norm_error_mps2;
        const double alpha =
            acceleration_is_valid ? 1.0 - std::exp(-dt_s / time_constant) : 0.0;
        gravityUpdate(
            alpha,
            orientation_,
            gravity_vector_,
            acceleration,
            &orientation_,
            &gravity_vector_);

        time_ns_ = interval_end_ns;
        if (*time_ns_ >= samples_[following].timestamp_ns) {
            following_index_ = following + 1;
        }
    }
    return orientation_;
}

void RawImuTracker::orientationsAt(
    const TimestampNs* timestamps_ns,
    std::size_t count,
    Eigen::Quaterniond* output_orientations) {
    if (count > 0 && (timestamps_ns == nullptr || output_orientations == nullptr)) {
        throw std::invalid_argument("batch IMU query buffers must be non-null");
    }
    for (std::size_t index = 1; index < count; ++index) {
        if (timestamps_ns[index] < timestamps_ns[index - 1]) {
            throw std::invalid_argument("IMU query timestamps must be sorted");
        }
    }
    for (std::size_t index = 0; index < count; ++index) {
        output_orientations[index] = advance(timestamps_ns[index]);
    }
}

std::vector<Eigen::Quaterniond> RawImuTracker::orientationsAt(
    const std::vector<TimestampNs>& timestamps_ns) {
    std::vector<Eigen::Quaterniond> output(timestamps_ns.size());
    orientationsAt(timestamps_ns.data(), timestamps_ns.size(), output.data());
    return output;
}

RawConstantVelocityPosePredictor::RawConstantVelocityPosePredictor(
    std::vector<ImuSample> samples,
    RigidPose initial_pose,
    RawImuTrackerOptions options)
    : tracker_(std::move(samples), options), initial_pose_(std::move(initial_pose)) {
    validatePose(initial_pose_);
}

void RawConstantVelocityPosePredictor::rememberInitialOrientation(
    const Eigen::Quaterniond& orientation) {
    if (!has_initial_tracker_orientation_) {
        initial_tracker_orientation_ = orientation;
        has_initial_tracker_orientation_ = true;
    }
}

Eigen::Vector3d RawConstantVelocityPosePredictor::velocity() const {
    if (correction_count_ < 2) {
        return Eigen::Vector3d::Zero();
    }
    const Correction& previous = corrections_[0];
    const Correction& anchor = corrections_[1];
    const double dt_s =
        static_cast<double>(anchor.timestamp_ns - previous.timestamp_ns) * 1.0e-9;
    return (anchor.pose.translation - previous.pose.translation) / dt_s;
}

RigidPose RawConstantVelocityPosePredictor::predictionFromTrackerQuaternion(
    TimestampNs timestamp_ns,
    const Eigen::Quaterniond& tracker_orientation) const {
    if (correction_count_ > 0) {
        const Correction& anchor = corrections_[correction_count_ - 1];
        if (timestamp_ns == anchor.timestamp_ns &&
            equalCoefficients(tracker_orientation, anchor.tracker_orientation)) {
            return anchor.pose;
        }
        const Eigen::Quaterniond delta = binaryQuaternionProduct(
            binaryQuaternionInverse(anchor.tracker_orientation), tracker_orientation);
        Eigen::Vector3d translation = anchor.pose.translation;
        if (correction_count_ == 2) {
            const Correction& previous = corrections_[0];
            translation = predictTranslation(
                previous.timestamp_ns,
                previous.pose.translation,
                anchor.timestamp_ns,
                anchor.pose.translation,
                anchor.pose.rotation,
                timestamp_ns);
        }
        return RigidPose{
            translation,
            binaryQuaternionProduct(anchor.pose.rotation, delta)};
    }
    if (!has_initial_tracker_orientation_) {
        throw std::logic_error("the pose predictor has not initialized its IMU tracker");
    }
    const Eigen::Quaterniond delta = binaryQuaternionProduct(
        binaryQuaternionInverse(initial_tracker_orientation_), tracker_orientation);
    return RigidPose{
        initial_pose_.translation,
        binaryQuaternionProduct(initial_pose_.rotation, delta)};
}

RigidPose RawConstantVelocityPosePredictor::predict(TimestampNs timestamp_ns) {
    const Eigen::Quaterniond& tracker_orientation = tracker_.advance(timestamp_ns);
    rememberInitialOrientation(tracker_orientation);
    return predictionFromTrackerQuaternion(timestamp_ns, tracker_orientation);
}

void RawConstantVelocityPosePredictor::correct(
    TimestampNs timestamp_ns,
    const RigidPose& pose) {
    validatePose(pose);
    const Eigen::Quaterniond& tracker_orientation = tracker_.advance(timestamp_ns);
    rememberInitialOrientation(tracker_orientation);
    if (correction_count_ > 0 &&
        timestamp_ns < corrections_[correction_count_ - 1].timestamp_ns) {
        throw std::invalid_argument("pose corrections must be time ordered");
    }
    const Correction correction{timestamp_ns, pose, tracker_orientation};
    if (correction_count_ > 0 &&
        timestamp_ns == corrections_[correction_count_ - 1].timestamp_ns) {
        corrections_[correction_count_ - 1] = correction;
    } else if (correction_count_ < 2) {
        corrections_[correction_count_++] = correction;
    } else {
        corrections_[0] = corrections_[1];
        corrections_[1] = correction;
    }
}

void RawConstantVelocityPosePredictor::reserveRayScratch(std::size_t ray_count) {
    unique_times_scratch_.reserve(ray_count + 1);
    orientation_scratch_.reserve(ray_count + 1);
}

void RawConstantVelocityPosePredictor::relativeMotion(
    const TimestampNs* point_timestamps_ns,
    std::size_t point_count,
    TimestampNs end_timestamp_ns,
    Eigen::Quaterniond* output_end_from_ray_rotations,
    Eigen::Vector3d* output_end_from_ray_translations) {
    if (point_count == 0) {
        throw std::invalid_argument("relative motion needs at least one ray timestamp");
    }
    if (point_timestamps_ns == nullptr || output_end_from_ray_rotations == nullptr ||
        output_end_from_ray_translations == nullptr) {
        throw std::invalid_argument("relative motion buffers must be non-null");
    }
    unique_times_scratch_.assign(
        point_timestamps_ns, point_timestamps_ns + point_count);
    for (std::size_t index = 0; index < point_count; ++index) {
        if (point_timestamps_ns[index] > end_timestamp_ns) {
            throw std::invalid_argument("ray timestamp exceeds the batch end");
        }
    }
    unique_times_scratch_.push_back(end_timestamp_ns);
    std::sort(unique_times_scratch_.begin(), unique_times_scratch_.end());
    unique_times_scratch_.erase(
        std::unique(unique_times_scratch_.begin(), unique_times_scratch_.end()),
        unique_times_scratch_.end());
    orientation_scratch_.resize(unique_times_scratch_.size());
    tracker_.orientationsAt(
        unique_times_scratch_.data(),
        unique_times_scratch_.size(),
        orientation_scratch_.data());
    rememberInitialOrientation(orientation_scratch_.front());

    const auto orientationAt = [this](TimestampNs timestamp_ns) -> const Eigen::Quaterniond& {
        const auto iterator = std::lower_bound(
            unique_times_scratch_.begin(), unique_times_scratch_.end(), timestamp_ns);
        const std::size_t index =
            static_cast<std::size_t>(iterator - unique_times_scratch_.begin());
        return orientation_scratch_[index];
    };
    const Eigen::Quaterniond& end_orientation = orientationAt(end_timestamp_ns);
    const Eigen::Quaterniond inverse_end = binaryQuaternionInverse(end_orientation);
    for (std::size_t index = 0; index < point_count; ++index) {
        output_end_from_ray_rotations[index] =
            arrayQuaternionProduct(inverse_end, orientationAt(point_timestamps_ns[index]));
    }

    const Eigen::Quaterniond& first_point_orientation = orientationAt(point_timestamps_ns[0]);
    const RigidPose predicted_start = predictionFromTrackerQuaternion(
        point_timestamps_ns[0], first_point_orientation);
    const RigidPose predicted_end = predictionFromTrackerQuaternion(
        end_timestamp_ns, end_orientation);
    last_end_from_start_pose_ =
        endFromStartPoseWithRoundtrip(predicted_start, predicted_end);
    has_last_end_from_start_pose_ = true;

    // scipy Rotation, used by the reference predictor here, normalizes stored coefficients.
    const Eigen::Quaterniond normalized_end = predicted_end.rotation.normalized();
    const Eigen::Vector3d velocity_in_end = normalized_end.inverse() * velocity();
    for (std::size_t index = 0; index < point_count; ++index) {
        const double relative_seconds =
            static_cast<double>(point_timestamps_ns[index] - end_timestamp_ns) * 1.0e-9;
        output_end_from_ray_translations[index] = relative_seconds * velocity_in_end;
    }
}

RelativeMotionBatch RawConstantVelocityPosePredictor::relativeMotion(
    const std::vector<TimestampNs>& point_timestamps_ns,
    TimestampNs end_timestamp_ns) {
    RelativeMotionBatch output;
    output.rotations.resize(point_timestamps_ns.size());
    output.translations.resize(point_timestamps_ns.size());
    relativeMotion(
        point_timestamps_ns.data(),
        point_timestamps_ns.size(),
        end_timestamp_ns,
        output.rotations.data(),
        output.translations.data());
    return output;
}

const RigidPose& RawConstantVelocityPosePredictor::lastEndFromStartPose() const {
    if (!has_last_end_from_start_pose_) {
        throw std::logic_error("relativeMotion has not produced a range pose");
    }
    return last_end_from_start_pose_;
}

RawImuTracker& RawConstantVelocityPosePredictor::tracker() noexcept {
    return tracker_;
}

const RawImuTracker& RawConstantVelocityPosePredictor::tracker() const noexcept {
    return tracker_;
}

}  // namespace navvis_recon::slam
