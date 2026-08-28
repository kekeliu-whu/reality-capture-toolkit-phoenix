#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <vector>

namespace navvis_recon::slam {

using TimestampNs = std::int64_t;

struct ImuSample {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TimestampNs timestamp_ns = 0;
    Eigen::Vector3d linear_acceleration = Eigen::Vector3d::Zero();
    Eigen::Vector3d angular_velocity = Eigen::Vector3d::Zero();
    // Firmware coefficients are preserved verbatim; they need not be exactly unit length.
    Eigen::Quaterniond orientation = Eigen::Quaterniond::Identity();
};

struct RigidPose {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    // The raw coefficient norm and sign are part of the reconstructed frontend state.
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();

    [[nodiscard]] static RigidPose identity() noexcept;
};

struct RawImuTrackerOptions {
    double initial_gravity_time_constant_s = 2.0;
    double steady_state_gravity_time_constant_s = 20.0;
    double time_constant_init_duration_s = 4.0;
    double time_constant_fade_duration_s = 40.0;
    double max_gravity_norm_error_mps2 = 4.0;
    double gravity_mps2 = 9.81;
    bool init_tilt_from_imu_orientation = true;
};

[[nodiscard]] double secondsFromNanoseconds(TimestampNs duration_ns) noexcept;

class RawImuTracker {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit RawImuTracker(
        std::vector<ImuSample> samples,
        RawImuTrackerOptions options = RawImuTrackerOptions{});

    [[nodiscard]] bool initialized() const noexcept;
    [[nodiscard]] std::optional<TimestampNs> timeNs() const noexcept;
    [[nodiscard]] const Eigen::Quaterniond& orientation() const;
    [[nodiscard]] Eigen::Vector3d gravityObservation() const;

    // Queries are stateful and must be nondecreasing across calls.
    [[nodiscard]] const Eigen::Quaterniond& advance(TimestampNs timestamp_ns);

    // No-allocation batch API. The input must be sorted in nondecreasing order.
    void orientationsAt(
        const TimestampNs* timestamps_ns,
        std::size_t count,
        Eigen::Quaterniond* output_orientations);

    // Convenience overload; this performs one batch allocation for its return value.
    [[nodiscard]] std::vector<Eigen::Quaterniond> orientationsAt(
        const std::vector<TimestampNs>& timestamps_ns);

private:
    struct Bracket {
        std::size_t low = 0;
        std::size_t high = 1;
    };

    [[nodiscard]] Bracket bracket(TimestampNs timestamp_ns) const;
    [[nodiscard]] Eigen::Quaterniond interpolateOrientation(TimestampNs timestamp_ns) const;
    [[nodiscard]] double gravityTimeConstant(double elapsed_s) const noexcept;

    std::vector<ImuSample> samples_;
    RawImuTrackerOptions options_;
    std::optional<TimestampNs> time_ns_;
    TimestampNs initial_time_ns_ = 0;
    Eigen::Quaterniond orientation_ = Eigen::Quaterniond::Identity();
    Eigen::Vector3d gravity_vector_ = Eigen::Vector3d::Zero();
    std::size_t following_index_ = 0;
};

struct RelativeMotionBatch {
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    std::vector<Eigen::Quaterniond> rotations;
    std::vector<Eigen::Vector3d> translations;
};

class RawConstantVelocityPosePredictor {
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    explicit RawConstantVelocityPosePredictor(
        std::vector<ImuSample> samples,
        RigidPose initial_pose = RigidPose::identity(),
        RawImuTrackerOptions options = RawImuTrackerOptions{});

    [[nodiscard]] RigidPose predict(TimestampNs timestamp_ns);
    void correct(TimestampNs timestamp_ns, const RigidPose& pose);

    // Reserves reusable internal sorting/orientation buffers for a ray batch.
    void reserveRayScratch(std::size_t ray_count);

    // No-allocation output API after reserveRayScratch() has established capacity.
    // Point timestamps may be unsorted or repeated, but none may exceed end_timestamp_ns.
    void relativeMotion(
        const TimestampNs* point_timestamps_ns,
        std::size_t point_count,
        TimestampNs end_timestamp_ns,
        Eigen::Quaterniond* output_end_from_ray_rotations,
        Eigen::Vector3d* output_end_from_ray_translations);

    // Convenience overload; this allocates the two returned batch arrays.
    [[nodiscard]] RelativeMotionBatch relativeMotion(
        const std::vector<TimestampNs>& point_timestamps_ns,
        TimestampNs end_timestamp_ns);

    [[nodiscard]] const RigidPose& lastEndFromStartPose() const;
    [[nodiscard]] RawImuTracker& tracker() noexcept;
    [[nodiscard]] const RawImuTracker& tracker() const noexcept;

private:
    struct Correction {
        EIGEN_MAKE_ALIGNED_OPERATOR_NEW

        TimestampNs timestamp_ns = 0;
        RigidPose pose;
        Eigen::Quaterniond tracker_orientation = Eigen::Quaterniond::Identity();
    };

    void rememberInitialOrientation(const Eigen::Quaterniond& orientation);
    [[nodiscard]] Eigen::Vector3d velocity() const;
    [[nodiscard]] RigidPose predictionFromTrackerQuaternion(
        TimestampNs timestamp_ns,
        const Eigen::Quaterniond& tracker_orientation) const;

    RawImuTracker tracker_;
    RigidPose initial_pose_;
    bool has_initial_tracker_orientation_ = false;
    Eigen::Quaterniond initial_tracker_orientation_ = Eigen::Quaterniond::Identity();
    std::array<Correction, 2> corrections_{};
    std::size_t correction_count_ = 0;
    bool has_last_end_from_start_pose_ = false;
    RigidPose last_end_from_start_pose_;

    // Reused across relativeMotion() calls; capacity grows only when a larger batch arrives.
    std::vector<TimestampNs> unique_times_scratch_;
    std::vector<Eigen::Quaterniond> orientation_scratch_;
};

}  // namespace navvis_recon::slam
