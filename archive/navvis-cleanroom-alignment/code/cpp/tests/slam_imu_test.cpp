#include "navvis_recon/slam_imu.hpp"

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <array>
#include <cmath>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

using navvis_recon::slam::ImuSample;
using navvis_recon::slam::RawConstantVelocityPosePredictor;
using navvis_recon::slam::RawImuTracker;
using navvis_recon::slam::RigidPose;
using navvis_recon::slam::TimestampNs;

void requireNear(double actual, double expected, double tolerance, const std::string& label) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > tolerance) {
        throw std::runtime_error(
            label + ": expected " + std::to_string(expected) + ", got " +
            std::to_string(actual));
    }
}

void requireQuaternionNear(
    const Eigen::Quaterniond& actual,
    const Eigen::Quaterniond& expected,
    double tolerance,
    const std::string& label) {
    requireNear(actual.x(), expected.x(), tolerance, label + ".x");
    requireNear(actual.y(), expected.y(), tolerance, label + ".y");
    requireNear(actual.z(), expected.z(), tolerance, label + ".z");
    requireNear(actual.w(), expected.w(), tolerance, label + ".w");
}

void requireVectorNear(
    const Eigen::Vector3d& actual,
    const Eigen::Vector3d& expected,
    double tolerance,
    const std::string& label) {
    for (int axis = 0; axis < 3; ++axis) {
        requireNear(actual[axis], expected[axis], tolerance, label + "." + std::to_string(axis));
    }
}

std::vector<ImuSample> constantYawSamples() {
    std::vector<ImuSample> samples;
    for (const TimestampNs timestamp : {0LL, 1'000'000'000LL, 2'000'000'000LL}) {
        samples.push_back(ImuSample{
            timestamp,
            Eigen::Vector3d(0.0, 0.0, 9.81),
            Eigen::Vector3d(0.0, 0.0, 0.1),
            Eigen::Quaterniond::Identity()});
    }
    return samples;
}

std::vector<ImuSample> variableSamples() {
    const std::array<TimestampNs, 6> timestamps{
        1'000'000'000LL,
        1'010'000'000LL,
        1'023'000'000LL,
        1'041'000'000LL,
        1'062'000'000LL,
        1'090'000'000LL};
    const std::array<Eigen::Vector3d, 6> accelerations{{
        {0.05, -0.02, 9.80},
        {0.08, -0.01, 9.79},
        {0.04, 0.03, 9.83},
        {-0.02, 0.05, 9.81},
        {-0.06, 0.02, 9.78},
        {-0.03, -0.04, 9.82},
    }};
    const std::array<Eigen::Vector3d, 6> angular_velocities{{
        {0.012, -0.018, 0.081},
        {0.015, -0.014, 0.086},
        {0.019, -0.009, 0.078},
        {0.011, -0.004, 0.073},
        {0.006, 0.002, 0.069},
        {0.002, 0.008, 0.075},
    }};
    const std::array<Eigen::Quaterniond, 6> orientations{{
        {0.9991, 0.0101, -0.0202, 0.0303},
        {0.9989, 0.0103, -0.0201, 0.0308},
        {0.9987, 0.0107, -0.0198, 0.0312},
        {0.9984, 0.0110, -0.0192, 0.0319},
        {0.9981, 0.0113, -0.0187, 0.0325},
        {0.9978, 0.0115, -0.0181, 0.0331},
    }};
    std::vector<ImuSample> samples;
    samples.reserve(timestamps.size());
    for (std::size_t index = 0; index < timestamps.size(); ++index) {
        samples.push_back(ImuSample{
            timestamps[index],
            accelerations[index],
            angular_velocities[index],
            orientations[index]});
    }
    return samples;
}

void testFirmwareQuaternionIsPreserved() {
    const Eigen::Quaterniond raw(
        0.9 * 1.00000004,
        0.1 * 1.00000004,
        -0.2 * 1.00000004,
        0.3 * 1.00000004);
    std::vector<ImuSample> samples{
        {0, Eigen::Vector3d(0.0, 0.0, 9.81), Eigen::Vector3d::Zero(), raw},
        {1'000'000'000LL,
         Eigen::Vector3d(0.0, 0.0, 9.81),
         Eigen::Vector3d::Zero(),
         raw}};
    RawImuTracker tracker(std::move(samples));
    requireQuaternionNear(tracker.advance(0), raw, 0.0, "firmware_orientation");
}

void testConstantYawBatch() {
    RawImuTracker tracker(constantYawSamples());
    const std::vector<TimestampNs> timestamps{0, 500'000'000LL, 1'000'000'000LL};
    const std::vector<Eigen::Quaterniond> orientations = tracker.orientationsAt(timestamps);
    const std::array<double, 3> angles{0.0, 0.05, 0.1};
    for (std::size_t index = 0; index < orientations.size(); ++index) {
        const Eigen::Quaterniond expected(
            Eigen::AngleAxisd(angles[index], Eigen::Vector3d::UnitZ()));
        requireQuaternionNear(orientations[index], expected, 2.0e-15, "constant_yaw");
    }
    requireVectorNear(
        tracker.gravityObservation(), Eigen::Vector3d(0.0, 0.0, 9.81), 2.0e-14,
        "gravity_observation");
}

void testConstantVelocityRelativeMotion() {
    RawConstantVelocityPosePredictor predictor(constantYawSamples());
    predictor.reserveRayScratch(4);
    predictor.correct(0, RigidPose::identity());
    predictor.correct(
        1'000'000'000LL,
        RigidPose{Eigen::Vector3d(1.0, 0.0, 0.0), Eigen::Quaterniond::Identity()});

    const std::vector<TimestampNs> point_times{
        1'250'000'000LL, 1'500'000'000LL, 1'250'000'000LL};
    const auto motion = predictor.relativeMotion(point_times, 1'500'000'000LL);
    requireVectorNear(
        motion.translations[0],
        Eigen::Vector3d(-0.24968756509874160, 0.012494792317669588, 0.0),
        2.0e-14, "relative_translation_0");
    requireVectorNear(motion.translations[1], Eigen::Vector3d::Zero(), 2.0e-14,
                      "relative_translation_1");
    requireVectorNear(motion.translations[2], motion.translations[0], 0.0,
                      "duplicate_relative_translation");

    const Eigen::Quaterniond expected_quarter_yaw(
        Eigen::AngleAxisd(-0.025, Eigen::Vector3d::UnitZ()));
    requireQuaternionNear(motion.rotations[0], expected_quarter_yaw, 2.0e-15,
                          "relative_rotation_0");
    requireQuaternionNear(motion.rotations[1], Eigen::Quaterniond::Identity(), 2.0e-15,
                          "relative_rotation_1");

    const RigidPose& end_from_start = predictor.lastEndFromStartPose();
    requireVectorNear(
        end_from_start.translation,
        Eigen::Vector3d(-0.24968756509874135, 0.012494792317669624, 0.0),
        3.0e-14,
        "end_from_start_translation");

    std::cout << std::setprecision(17);
    std::cout << "orientation_500ms_xyzw=" << 0.0 << ',' << 0.0 << ','
              << std::sin(0.025) << ',' << std::cos(0.025) << '\n';
    std::cout << "relative_rotation_1250ms_xyzw=" << motion.rotations[0].x() << ','
              << motion.rotations[0].y() << ',' << motion.rotations[0].z() << ','
              << motion.rotations[0].w() << '\n';
    std::cout << "relative_translation_1250ms_xyz=" << motion.translations[0].transpose()
              << '\n';
    std::cout << "end_from_start_translation_xyz=" << end_from_start.translation.transpose()
              << '\n';
}

void testVariableSamplesAgainstReferenceKeys() {
    RawImuTracker tracker(variableSamples());
    const std::vector<TimestampNs> query_times{
        1'003'000'000LL,
        1'008'000'000LL,
        1'019'000'000LL,
        1'041'000'000LL,
        1'079'000'000LL};
    const auto orientations = tracker.orientationsAt(query_times);
    const std::array<Eigen::Quaterniond, 5> expected_orientations{{
        {0.99903999999999993, 0.010159999999999999, -0.02017, 0.030450000000000001},
        {0.99927382403273857, 0.010176080810089579, -0.020194227342545962,
         0.030667004489705916},
        {0.99925950803982344, 0.010210025256185924, -0.020180256701826216,
         0.031127933046169128},
        {0.99923522932575193, 0.010279720309633795, -0.020034003696980954,
         0.031967522902739846},
        {0.99920077822579645, 0.010189162499622487, -0.01959573978298261,
         0.03331655358642021},
    }};
    for (std::size_t index = 0; index < orientations.size(); ++index) {
        requireQuaternionNear(
            orientations[index], expected_orientations[index], 4.0e-15,
            "variable_orientation_" + std::to_string(index));
    }
    const Eigen::Vector3d expected_gravity(
        0.39082149682631218, 0.18694243218880435, 9.8004291275773117);
    requireVectorNear(tracker.gravityObservation(), expected_gravity, 4.0e-14,
                      "variable_gravity");

    RawConstantVelocityPosePredictor predictor(variableSamples());
    predictor.reserveRayScratch(4);
    predictor.correct(
        1'003'000'000LL,
        RigidPose{
            Eigen::Vector3d(0.2, -0.1, 0.05),
            Eigen::Quaterniond(0.999, 0.01, -0.02, 0.03)});
    predictor.correct(
        1'041'000'000LL,
        RigidPose{
            Eigen::Vector3d(0.24, -0.115, 0.052),
            Eigen::Quaterniond(0.9989, 0.012, -0.018, 0.034)});
    const std::vector<TimestampNs> point_times{
        1'047'000'000LL, 1'071'000'000LL, 1'055'000'000LL, 1'079'000'000LL};
    const auto motion = predictor.relativeMotion(point_times, 1'079'000'000LL);
    const std::array<Eigen::Quaterniond, 4> expected_rotations{{
        {0.99999928416085559, 4.796502227585593e-05, -0.00039285990965761872,
         -0.001129175905077647},
        {0.99999995269467856, 2.5024608964286125e-05, -0.00010746551912116305,
         -0.00028711595589874146},
        {0.99999959440125896, 5.002717341832648e-05, -0.00030708361432703411,
         -0.00084521846496439003},
        {1.0, 0.0, 0.0, 0.0},
    }};
    const std::array<Eigen::Vector3d, 4> expected_translations{{
        {-0.032752503485017757, 0.014952562025177385, -0.0008448638522506127},
        {-0.0081881258712544393, 0.0037381405062943463, -0.00021121596306265317},
        {-0.024564377613763316, 0.011214421518883039, -0.00063364788918795947},
        {0.0, 0.0, 0.0},
    }};
    for (std::size_t index = 0; index < point_times.size(); ++index) {
        requireQuaternionNear(
            motion.rotations[index], expected_rotations[index], 5.0e-15,
            "variable_relative_rotation_" + std::to_string(index));
        requireVectorNear(
            motion.translations[index], expected_translations[index], 5.0e-14,
            "variable_relative_translation_" + std::to_string(index));
    }
    const RigidPose& end_from_start = predictor.lastEndFromStartPose();
    requireVectorNear(
        end_from_start.translation,
        Eigen::Vector3d(
            -0.032759394223924987, 0.014938900741499395, -0.0008496571161689323),
        5.0e-14,
        "variable_end_from_start_translation");
    requireQuaternionNear(
        end_from_start.rotation,
        Eigen::Quaterniond(
            0.99999928416085571,
            4.7965022275855279e-05,
            -0.00039285990965762219,
            -0.0011291759050776271),
        5.0e-15,
        "variable_end_from_start_rotation");

    std::cout << "variable_orientation_last_xyzw=" << orientations.back().x() << ','
              << orientations.back().y() << ',' << orientations.back().z() << ','
              << orientations.back().w() << '\n';
    std::cout << "variable_gravity_xyz=" << tracker.gravityObservation().transpose() << '\n';
}

}  // namespace

int main() {
    try {
        testFirmwareQuaternionIsPreserved();
        testConstantYawBatch();
        testConstantVelocityRelativeMotion();
        testVariableSamplesAgainstReferenceKeys();
        std::cout << "slam_imu_test=PASS\n";
        return EXIT_SUCCESS;
    } catch (const std::exception& error) {
        std::cerr << "slam_imu_test=FAIL: " << error.what() << '\n';
        return EXIT_FAILURE;
    }
}
