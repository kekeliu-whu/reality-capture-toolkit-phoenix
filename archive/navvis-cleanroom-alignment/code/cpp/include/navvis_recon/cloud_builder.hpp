#pragma once

#include "navvis_recon/types.hpp"

#include <functional>
#include <memory>
#include <unordered_map>

namespace navvis_recon {

class PointFilter {
public:
    virtual ~PointFilter() = default;
    [[nodiscard]] virtual bool keep(const LaserPoint& point) const = 0;
};

class NonFiniteXYZFilter final : public PointFilter {
public:
    [[nodiscard]] bool keep(const LaserPoint& point) const override;
};

class RingRangeFilter final : public PointFilter {
public:
    std::unordered_map<std::uint16_t, float> minimum_range;
    std::unordered_map<std::uint16_t, float> maximum_range;
    [[nodiscard]] bool keep(const LaserPoint& point) const override;
};

class RegionFilter final : public PointFilter {
public:
    std::vector<AxisAlignedRegion> rejected_regions;
    [[nodiscard]] bool keep(const LaserPoint& point) const override;
};

class IntensityRegionFilter final : public PointFilter {
public:
    float minimum_intensity = 0.05F;
    std::vector<AxisAlignedRegion> reflection_regions;
    [[nodiscard]] bool keep(const LaserPoint& point) const override;
};

class MultilayerFringeFilter final : public PointFilter {
public:
    std::vector<AxisAlignedRegion> fringe_regions;
    [[nodiscard]] bool keep(const LaserPoint& point) const override;
};

class PlaneFilter {
public:
    float distance_threshold = 0.02F;
    int iterations = 96;
    [[nodiscard]] std::vector<std::size_t> largestPlane(const std::vector<LaserPoint>& cloud) const;
};

struct CloudBuilderOptions {
    bool unskew = true;
    bool normalize_intensity = true;
    bool remove_no_motion_scans = false;
    float no_motion_translation_m = 0.005F;
    float no_motion_angle_deg = 0.05F;
    std::vector<std::shared_ptr<const PointFilter>> filters;
};

class IntensityNormalizer {
public:
    static void normalizePerRing(std::vector<LaserPoint>& cloud);
};

class CloudBuilder {
public:
    explicit CloudBuilder(CloudBuilderOptions options);

    [[nodiscard]] std::vector<LaserPoint> build(
        const std::vector<std::vector<LaserPoint>>& scans,
        const std::vector<Pose>& trajectory,
        const std::function<Vec3f(const LaserPoint&)>& sensor_to_rig = {}) const;

    static Pose poseAt(const std::vector<Pose>& trajectory, double timestamp);
    static void computeNormalsFromOrderedCloud(std::vector<std::vector<LaserPoint>>& rows);

private:
    [[nodiscard]] bool scanMoved(const Pose& previous, const Pose& current) const;
    CloudBuilderOptions options_;
};

}  // namespace navvis_recon

