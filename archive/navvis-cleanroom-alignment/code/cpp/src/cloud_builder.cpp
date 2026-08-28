#include "navvis_recon/cloud_builder.hpp"

#include <Eigen/Geometry>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <random>
#include <stdexcept>

namespace navvis_recon {

bool NonFiniteXYZFilter::keep(const LaserPoint& point) const {
    return point.xyz.array().isFinite().all();
}

bool RingRangeFilter::keep(const LaserPoint& point) const {
    const float distance = (point.xyz - point.origin).norm();
    const auto minimum = minimum_range.find(point.ring);
    const auto maximum = maximum_range.find(point.ring);
    const float lower = minimum == minimum_range.end() ? 0.0F : minimum->second;
    const float upper = maximum == maximum_range.end()
                            ? std::numeric_limits<float>::infinity()
                            : maximum->second;
    return distance >= lower && distance <= upper;
}

bool RegionFilter::keep(const LaserPoint& point) const {
    return std::none_of(rejected_regions.begin(), rejected_regions.end(),
                        [&](const AxisAlignedRegion& region) { return region.contains(point.xyz); });
}

bool IntensityRegionFilter::keep(const LaserPoint& point) const {
    if (point.intensity >= minimum_intensity) {
        return true;
    }
    return std::none_of(reflection_regions.begin(), reflection_regions.end(),
                        [&](const AxisAlignedRegion& region) { return region.contains(point.xyz); });
}

bool MultilayerFringeFilter::keep(const LaserPoint& point) const {
    return std::none_of(fringe_regions.begin(), fringe_regions.end(),
                        [&](const AxisAlignedRegion& region) { return region.contains(point.xyz); });
}

std::vector<std::size_t> PlaneFilter::largestPlane(const std::vector<LaserPoint>& cloud) const {
    if (cloud.size() < 3U) {
        return {};
    }
    std::mt19937 generator(0);
    std::uniform_int_distribution<std::size_t> sample(0, cloud.size() - 1);
    std::vector<std::size_t> best;
    for (int iteration = 0; iteration < iterations; ++iteration) {
        const auto ia = sample(generator);
        const auto ib = sample(generator);
        const auto ic = sample(generator);
        if (ia == ib || ia == ic || ib == ic) {
            continue;
        }
        const Vec3f normal = (cloud[ib].xyz - cloud[ia].xyz).cross(cloud[ic].xyz - cloud[ia].xyz);
        if (normal.squaredNorm() < 1.0e-10F) {
            continue;
        }
        const Vec3f unit_normal = normal.normalized();
        std::vector<std::size_t> inliers;
        for (std::size_t i = 0; i < cloud.size(); ++i) {
            if (std::abs(unit_normal.dot(cloud[i].xyz - cloud[ia].xyz)) <= distance_threshold) {
                inliers.push_back(i);
            }
        }
        if (inliers.size() > best.size()) {
            best = std::move(inliers);
        }
    }
    return best;
}

void IntensityNormalizer::normalizePerRing(std::vector<LaserPoint>& cloud) {
    std::unordered_map<std::uint16_t, std::vector<std::size_t>> by_ring;
    for (std::size_t i = 0; i < cloud.size(); ++i) {
        by_ring[cloud[i].ring].push_back(i);
    }
    for (const auto& entry : by_ring) {
        std::vector<float> values;
        values.reserve(entry.second.size());
        for (const auto index : entry.second) {
            values.push_back(cloud[index].intensity);
        }
        std::sort(values.begin(), values.end());
        const auto percentile = [&](float fraction) {
            return values[static_cast<std::size_t>(fraction * static_cast<float>(values.size() - 1))];
        };
        const float low = percentile(0.05F);
        const float high = percentile(0.95F);
        const float span = std::max(high - low, 1.0e-6F);
        for (const auto index : entry.second) {
            cloud[index].intensity = std::clamp((cloud[index].intensity - low) / span, 0.0F, 1.0F);
        }
    }
}

CloudBuilder::CloudBuilder(CloudBuilderOptions options) : options_(std::move(options)) {}

Pose CloudBuilder::poseAt(const std::vector<Pose>& trajectory, double timestamp) {
    if (trajectory.empty()) {
        throw std::invalid_argument("trajectory is empty");
    }
    const auto upper = std::upper_bound(
        trajectory.begin(), trajectory.end(), timestamp,
        [](double time, const Pose& pose) { return time < pose.timestamp; });
    if (upper == trajectory.begin()) {
        return trajectory.front();
    }
    if (upper == trajectory.end()) {
        return trajectory.back();
    }
    return Pose::interpolate(*std::prev(upper), *upper, timestamp);
}

bool CloudBuilder::scanMoved(const Pose& previous, const Pose& current) const {
    const float distance = (current.translation - previous.translation).norm();
    const float angle = Eigen::AngleAxisf(previous.rotation.conjugate() * current.rotation).angle() *
                        180.0F / static_cast<float>(M_PI);
    return distance >= options_.no_motion_translation_m || angle >= options_.no_motion_angle_deg;
}

std::vector<LaserPoint> CloudBuilder::build(
    const std::vector<std::vector<LaserPoint>>& scans,
    const std::vector<Pose>& trajectory,
    const std::function<Vec3f(const LaserPoint&)>& sensor_to_rig) const {
    std::vector<LaserPoint> result;
    std::optional<Pose> previous_scan_pose;
    for (const auto& scan : scans) {
        if (scan.empty()) {
            continue;
        }
        const Pose reference_pose = poseAt(trajectory, scan.front().timestamp);
        if (options_.remove_no_motion_scans && previous_scan_pose &&
            !scanMoved(*previous_scan_pose, reference_pose)) {
            continue;
        }
        previous_scan_pose = reference_pose;
        for (const LaserPoint& raw : scan) {
            LaserPoint rig_point = raw;
            rig_point.xyz = sensor_to_rig ? sensor_to_rig(raw) : raw.xyz;
            bool accepted = true;
            for (const auto& filter : options_.filters) {
                if (!filter->keep(rig_point)) {
                    accepted = false;
                    break;
                }
            }
            if (!accepted) {
                continue;
            }
            const Pose pose = options_.unskew ? poseAt(trajectory, raw.timestamp) : reference_pose;
            rig_point.xyz = pose.apply(rig_point.xyz);
            rig_point.origin = pose.translation;
            if (rig_point.has_normal) {
                rig_point.normal = pose.rotation * rig_point.normal;
            }
            result.push_back(rig_point);
        }
    }
    if (options_.normalize_intensity && !result.empty()) {
        IntensityNormalizer::normalizePerRing(result);
    }
    return result;
}

void CloudBuilder::computeNormalsFromOrderedCloud(std::vector<std::vector<LaserPoint>>& rows) {
    if (rows.size() < 3U) {
        return;
    }
    for (std::size_t row = 1; row + 1 < rows.size(); ++row) {
        const std::size_t width = std::min({rows[row - 1].size(), rows[row].size(), rows[row + 1].size()});
        for (std::size_t column = 1; column + 1 < width; ++column) {
            const Vec3f horizontal = rows[row][column + 1].xyz - rows[row][column - 1].xyz;
            const Vec3f vertical = rows[row + 1][column].xyz - rows[row - 1][column].xyz;
            Vec3f normal = normalizedOr(horizontal.cross(vertical));
            if (normal.dot(rows[row][column].xyz - rows[row][column].origin) > 0.0F) {
                normal = -normal;
            }
            rows[row][column].normal = normal;
            rows[row][column].has_normal = normal.squaredNorm() > 0.5F;
        }
    }
}

}  // namespace navvis_recon
