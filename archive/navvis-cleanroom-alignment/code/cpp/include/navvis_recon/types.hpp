#pragma once

#include <Eigen/Core>
#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace navvis_recon {

using Vec2f = Eigen::Vector2f;
using Vec3f = Eigen::Vector3f;
using Mat3f = Eigen::Matrix3f;

inline Vec3f normalizedOr(const Vec3f& value, const Vec3f& fallback = Vec3f::Zero()) {
    const float norm = value.norm();
    return norm > 1.0e-9F ? value / norm : fallback;
}

struct Pose {
    double timestamp = 0.0;
    Vec3f translation = Vec3f::Zero();
    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
    // Some dataset consumers reconstruct the float rotation matrix through a
    // rotation-vector conversion rather than Quaternionf::toRotationMatrix().
    // Preserve that matrix when it is available; ordinary Pose users keep the
    // quaternion path through the default false flag.
    Mat3f rotation_matrix = Mat3f::Identity();
    bool has_rotation_matrix = false;
    Eigen::Vector3d translation_double = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation_double = Eigen::Quaterniond::Identity();
    Eigen::Matrix3d rotation_matrix_double = Eigen::Matrix3d::Identity();
    bool has_double_pose = false;

    [[nodiscard]] Mat3f rotationMatrix() const {
        return has_rotation_matrix ? rotation_matrix : rotation.toRotationMatrix();
    }

    [[nodiscard]] Vec3f apply(const Vec3f& point) const {
        return rotationMatrix() * point + translation;
    }

    [[nodiscard]] Vec3f inverseApply(const Vec3f& point) const {
        return rotationMatrix().transpose() * (point - translation);
    }

    static Pose interpolate(const Pose& first, const Pose& second, double query_time) {
        if (second.timestamp <= first.timestamp) {
            return first;
        }
        const float alpha = static_cast<float>(std::clamp(
            (query_time - first.timestamp) / (second.timestamp - first.timestamp), 0.0, 1.0));
        return Pose{
            query_time,
            (1.0F - alpha) * first.translation + alpha * second.translation,
            first.rotation.slerp(alpha, second.rotation).normalized()};
    }
};

struct LaserPoint {
    Vec3f xyz = Vec3f::Zero();
    double timestamp = 0.0;
    float intensity = 0.0F;
    std::uint16_t ring = 0;
    Vec3f origin = Vec3f::Zero();
    Vec3f normal = Vec3f::Zero();
    float ray_weight = 1.0F;
    bool has_normal = false;
};

struct ColoredPoint : LaserPoint {
    Eigen::Matrix<std::uint8_t, 3, 1> rgb = Eigen::Matrix<std::uint8_t, 3, 1>::Zero();
    std::uint8_t alpha = 255;
    bool has_color = false;
};

struct AxisAlignedRegion {
    Vec3f minimum = Vec3f::Constant(-std::numeric_limits<float>::infinity());
    Vec3f maximum = Vec3f::Constant(std::numeric_limits<float>::infinity());

    [[nodiscard]] bool contains(const Vec3f& point) const {
        return (point.array() >= minimum.array()).all() && (point.array() <= maximum.array()).all();
    }
};

struct Camera {
    int width = 0;
    int height = 0;
    float fx = 1.0F;
    float fy = 1.0F;
    float cx = 0.0F;
    float cy = 0.0F;
    Pose world_from_camera;
    float rolling_shutter_seconds = 0.0F;

    [[nodiscard]] Eigen::Vector3f project(const Vec3f& point_world) const {
        const Vec3f camera_point = world_from_camera.inverseApply(point_world);
        if (camera_point.z() <= 1.0e-8F) {
            return {std::numeric_limits<float>::quiet_NaN(),
                    std::numeric_limits<float>::quiet_NaN(), camera_point.z()};
        }
        return {fx * camera_point.x() / camera_point.z() + cx,
                fy * camera_point.y() / camera_point.z() + cy,
                camera_point.z()};
    }

    [[nodiscard]] bool inside(float u, float v, float margin = 0.0F) const {
        return u >= margin && v >= margin && u < static_cast<float>(width) - margin &&
               v < static_cast<float>(height) - margin;
    }
};

}  // namespace navvis_recon
