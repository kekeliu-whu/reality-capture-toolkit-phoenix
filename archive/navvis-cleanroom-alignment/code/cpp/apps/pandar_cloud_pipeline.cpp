#include "navvis_recon/cloud_builder.hpp"
#include "navvis_recon/cloud_surface_filter.hpp"
#include "navvis_recon/types.hpp"

#include <Eigen/Core>
#include <Eigen/Eigenvalues>
#include <Eigen/Geometry>

#include <boost/random/mersenne_twister.hpp>
#include <boost/random/uniform_int.hpp>

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <climits>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <optional>
#include <numeric>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;
using namespace navvis_recon;

namespace {

using SteadyClock = std::chrono::steady_clock;

double elapsedSeconds(const SteadyClock::time_point start) {
    return std::chrono::duration<double>(SteadyClock::now() - start).count();
}

struct CloudBuilderTiming {
    double decode = 0.0;
    double transform = 0.0;
    double order_and_normals = 0.0;
    double fringe = 0.0;
    double foot = 0.0;
    double emit = 0.0;
    double shard_partition = 0.0;
    double shard_write = 0.0;
    double retain = 0.0;
};

constexpr float kPi = 3.14159265358979323846F;
constexpr double kPiDouble = 3.14159265358979323846;
constexpr int kBlocks = 6;
constexpr int kRings = 32;
constexpr int kPandarAngleCount = 36000;
constexpr int kPacketBytes = 820;
constexpr int kBlockBytes = 2 + kRings * 4;
constexpr int kBodyOffset = 12;
constexpr float kXtmCoordinateCorrectionH = 0.0305F;

// libnavvis_ros_utilities::RegionYawRange<float>::contains computes a strict
// shortest-angle test around a stored center.  It deliberately promotes the
// fmod operation to double after subtracting the center in float.  Reusing
// this sequence is important for returns that lie exactly on a configured
// angular boundary.
bool regionYawRangeContains(
    const Vec3f& point, float center, float half_width) {
    const float yaw = ::atan2f(point.y(), point.x());
    const float delta = yaw - center;
    const double wrapped =
        ::fmod(static_cast<double>(::fabsf(delta)) + kPiDouble,
               2.0 * kPiDouble) - kPiDouble;
    const int sign = (delta > 0.0F) - (delta < 0.0F);
    const float distance = static_cast<float>(static_cast<double>(sign) * wrapped);
    return half_width > ::fabsf(distance);
}

// PandarGeneral does not evaluate trigonometric functions while decoding a
// packet. Init() builds single-precision lookup tables from a double-precision
// degree-to-radian conversion and CalcXTPointXYZIT indexes those tables with
// the packet azimuth in hundredths of a degree. Keeping that mixed-precision
// path matters at voxel boundaries: evaluating sin/cos directly from a float
// radian angle moves XTM returns by several micrometres.
struct PandarSdkTrigonometry {
    std::array<float, kPandarAngleCount> azimuth_sin{};
    std::array<float, kPandarAngleCount> azimuth_cos{};
    std::array<float, kRings> elevation_sin{};
    std::array<float, kRings> elevation_cos{};

    PandarSdkTrigonometry();
};

// PandarGeneral initializes the XTM elevation tables with two independent
// libm calls (sinf followed by cosf). GCC may otherwise fuse adjacent source
// calls into sincosf, whose sine result differs by one ULP for -13 degrees on
// the frozen glibc build. That ULP changes the strict G11 foot-region test.
// Keep the wrappers out of line so the generated path retains the two calls.
#if defined(__GNUC__)
__attribute__((noinline, noipa))
#endif
float pandarSdkElevationSin(float radians) {
    return ::sinf(radians);
}

#if defined(__GNUC__)
__attribute__((noinline, noipa))
#endif
float pandarSdkElevationCos(float radians) {
    return ::cosf(radians);
}

// Exact parameters embedded in cloud_builder's
// PlaneFilter<PointXYZNormalITR> used for the G11 vertical-foot region.
constexpr float kFootRegionRadiusSquared = 0.25F;
constexpr float kFootRansacDistance = 0.01F;
constexpr float kFootMaximumPlaneDistance = 0.02F;
constexpr int kFootRansacMaximumIterations = 50;

constexpr int kVelodynePacketBytes = 1206;
constexpr int kVelodyneBlocks = 12;
constexpr int kVelodyneBlockBytes = 100;
constexpr int kVelodyneRings = 16;
constexpr float kVelodyneDistanceUnit = 0.002F;
constexpr float kVelodyneBlockDurationMicroseconds = 110.592F;
constexpr float kVelodyneFiringDurationMicroseconds = 55.296F;
constexpr float kVelodyneChannelDurationMicroseconds = 2.304F;
constexpr std::array<float, kVelodyneRings> kVlp16ElevationDegrees{{
    -15.0F, 1.0F, -13.0F, 3.0F, -11.0F, 5.0F, -9.0F, 7.0F,
    -7.0F, 9.0F, -5.0F, 11.0F, -3.0F, 13.0F, -1.0F, 15.0F,
}};

// Exact device-specific AIC models copied from cloud_builder after it applies
// the G11 workstation/device certificates.  The installed binary converts
// these values to float before decoding points, so they are stored as float
// here as well.  Angles are radians and distances are metres.
struct PandarIntrinsicCalibration {
    std::array<float, kRings> range_bias;
    std::array<float, kRings> cone_correction;
    std::array<float, kRings> azimuth_correction;
    float excentricity_amplitude;
    float excentricity_phase;
    float ray_parallax_amplitude;
    float ray_parallax_phase;
};

constexpr std::array<float, kRings> kNominalElevationDegrees{{
    19.5F, 18.2F, 16.9F, 15.6F, 14.3F, 13.0F, 11.7F, 10.4F,
    9.1F, 7.8F, 6.5F, 5.2F, 3.9F, 2.6F, 1.3F, 0.0F,
    -1.3F, -2.6F, -3.9F, -5.2F, -6.5F, -7.8F, -9.1F, -10.4F,
    -11.7F, -13.0F, -14.3F, -15.6F, -16.9F, -18.2F, -19.5F, -20.8F,
}};

PandarSdkTrigonometry::PandarSdkTrigonometry() {
    for (int index = 0; index < kPandarAngleCount; ++index) {
        const double degrees = static_cast<double>(index) * 0.01;
        const float radians = static_cast<float>(degrees * kPiDouble / 180.0);
        ::sincosf(radians, &azimuth_sin[index], &azimuth_cos[index]);
    }
    for (int ring = 0; ring < kRings; ++ring) {
        const float radians = static_cast<float>(
            static_cast<double>(kNominalElevationDegrees[ring]) * kPiDouble / 180.0);
        elevation_sin[ring] = pandarSdkElevationSin(radians);
        elevation_cos[ring] = pandarSdkElevationCos(radians);
    }
}

const PandarSdkTrigonometry& pandarSdkTrigonometry() {
    static const PandarSdkTrigonometry tables;
    return tables;
}

constexpr PandarIntrinsicCalibration kHorizontalIntrinsic{
    {{
        -0.0064009903F, -0.0044083981F, 0.00017022328393068165F,
        -0.001968690427020192F, -0.00167003960814327F, -0.0032098696F,
        0.00035861885407939553F, -0.0034141073F,
        -0.0010987194254994392F, -0.0033663870F, -0.0030978686F, -0.0017181244F,
        0.0024614704F, 0.0005311513F, -0.0065928494F, -0.0050865839F,
        0.0018090290250256658F, -0.0024438889F, -0.0072587920F, -0.0104295864F,
        -0.0034129131F, -0.0058908523F, -0.0097121110F, -0.0088721662F,
        -0.0048301369F, -0.0084977660F, -0.0077855905F, -0.0038469603F,
        -0.0070443515F, -0.0080129299F, -0.0049244263F, -0.002674720250070095F,
    }},
    {{
        -0.0042838064F, -0.0042006997F, -0.0042754564F, -0.0041130742F,
        -0.0039931032F, -0.0039446691F, -0.0031542611F, -0.0031620464F,
        -0.0030996611F, -0.0029984389F, -0.0030198139F, -0.0029844660F,
        -0.0030204249F, -0.0029894763F, -0.0029510286F, -0.0030397172F,
        -0.0030947318F, -0.0030171070247888565F, -0.0029433545F, -0.0028782960F,
        -0.0026685664F, -0.0026417128F, -0.0024279478F, -0.0024646779F,
        -0.002477211644873023F, -0.0018528641667217016F,
        -0.001700049266219139F, -0.0016997599F,
        -0.0016463762F, -0.0015596584F, -0.0014120611594989896F, -0.0012987547F,
    }},
    {{
        0.0023653494F, 0.0020340674F, 0.0016973834F, 0.0013188698F,
        0.0009686058619990945F, 0.000624827342107892F,
        0.000276800972642377F, -8.356367470696568e-05F,
        -0.0005034874F, -0.0008474150F, -0.0012137609F, -0.0016300334F,
        -0.0019664942F, -0.0023620331F, -0.0026969727F, -0.0030270637F,
        0.0023727087F, 0.0020648190F, 0.0017900909F, 0.0015249726F,
        0.0012494086F, 0.0009939138F, 0.0007668555481359363F,
        0.00045096094254404306F, 0.0002174649853259325F,
        3.083750925725326e-05F, -0.0002542381F, -0.0005930768F,
        -0.0008930586F, -0.0012389210751280189F, -0.0015528417F, -0.0018849763F,
    }},
    0.0003614433517213911F,
    1.2487449647F,
    0.033155F,
    0.0637192344F,
};

constexpr PandarIntrinsicCalibration kVerticalIntrinsic{
    {{
        0.0045513932F, -0.0028439953F, -0.0030504468F, -0.0014787343F,
        0.0038787856F, 0.0019287150F, -0.004449254367500544F, -0.0035124469F,
        0.0046473371F, 0.0038794594F, -0.0041709617F, -0.0065905286F,
        -0.0010690490F, 0.0005486645F, 0.0016900145F, -0.0014606433F,
        -0.0055200630F, -0.0027996888384222984F, 9.33080809772946e-05F,
        0.0006283001F,
        -0.0089430024F, -0.0042470453F, 0.0036448716F, 0.0030769677F,
        -0.0059297695F, -0.0023555987F, 0.0072759592F, 0.0061325571F,
        -0.0015874653F, 0.0036836546F, 0.0012476898F, 0.0012045690F,
    }},
    {{
        -0.0027557157F, -0.0028093945F, -0.0028683840F, -0.0029126871F,
        -0.0029378635F, -0.0029685604F, -0.0022423219F, -0.0023061121F,
        -0.0023255634F, -0.0023781506F, -0.0024278059F, -0.0024948161F,
        -0.0024127232F, -0.002479325979948044F, -0.0024377835F, -0.0025988791F,
        -0.00282728997990489F, -0.0027284078F, -0.0026377546F, -0.0024859116F,
        -0.0023476579F, -0.0022602282F, -0.0022156484F, -0.0022593747F,
        -0.0022958376F, -0.0016856953F, -0.0013645916F, -0.0013924642F,
        -0.0014819361F, -0.0014020747F, -0.00127972976770252F, -0.0011000842F,
    }},
    {{
        0.0018589229F, 0.0016094418F, 0.0013603715F, 0.0010415864F,
        0.0007438157335855067F, 0.0004132380709052086F,
        7.500357605749741e-05F, -0.0002517913526389748F,
        -0.0006132746348157525F, -0.0010217470F, -0.0013289616F, -0.0016833709F,
        -0.0020389269F, -0.0024163609F, -0.0027160347F, -0.0030894359F,
        0.0024053396F, 0.0021207807F, 0.0018318638F, 0.0015938593F,
        0.0013478107284754515F, 0.0011195164F, 0.0008478923F,
        0.0006578510510735214F, 0.0003892848617397249F,
        0.00016850657993927598F, -7.733784150332212e-05F,
        -0.00031324164592660964F,
        -0.0005690709F, -0.0008118788F, -0.0011640568F, -0.0014895953F,
    }},
    0.0006530820F,
    -0.6482042789F,
    0.033155F,
    0.0235000132F,
};

#include "pandar_xtm_g11_range_splines.inc"

float applyRangeBiasSpline(
    float range, const std::array<float, 6>& spline_values) {
    if (!std::isfinite(range)) {
        return range;
    }
    if (range <= kRangeSplineKnots.front()) {
        return range + spline_values.front();
    }
    for (std::size_t i = 0; i + 1U < kRangeSplineKnots.size(); ++i) {
        if (range <= kRangeSplineKnots[i + 1U]) {
            const float slope =
                (spline_values[i + 1U] - spline_values[i]) /
                (kRangeSplineKnots[i + 1U] - kRangeSplineKnots[i]);
            const float correction = spline_values[i] +
                (range - kRangeSplineKnots[i]) * slope;
            return range + correction;
        }
    }
    return range + spline_values.back();
}

constexpr std::array<float, kRings> kLaserTimeOffsetMicroseconds{{
    0.368F, 3.224F, 6.080F, 8.936F, 11.792F, 14.648F, 17.504F, 20.360F,
    23.216F, 26.072F, 28.928F, 31.784F, 34.640F, 37.496F, 40.352F, 43.208F,
    0.368F, 3.224F, 6.080F, 8.936F, 11.792F, 14.648F, 17.504F, 20.360F,
    23.216F, 26.072F, 28.928F, 31.784F, 34.640F, 37.496F, 40.352F, 43.208F,
}};

struct PrecisePose {
    double timestamp = 0.0;
    std::int64_t timestamp_us = 0;
    Eigen::Vector3d translation = Eigen::Vector3d::Zero();
    Eigen::Quaterniond rotation = Eigen::Quaterniond::Identity();
    // The trajectory provider stores the incoming quaternion before the
    // normalized pose used by the no-motion and VLP16 paths. CloudBuilder's
    // G11 transform provider first rounds these coefficients to float.
    Eigen::Quaterniond provider_rotation = Eigen::Quaterniond::Identity();

    [[nodiscard]] Eigen::Vector3d apply(const Eigen::Vector3d& point) const {
        return rotation * point + translation;
    }

    [[nodiscard]] Eigen::Vector3d inverseApply(const Eigen::Vector3d& point) const {
        return rotation.conjugate() * (point - translation);
    }

    static PrecisePose interpolate(
        const PrecisePose& first, const PrecisePose& second, double query_time) {
        if (second.timestamp <= first.timestamp) {
            return first;
        }
        const double alpha = std::clamp(
            (query_time - first.timestamp) / (second.timestamp - first.timestamp), 0.0, 1.0);
        return PrecisePose{
            query_time,
            static_cast<std::int64_t>(query_time * 1.0e6),
            (1.0 - alpha) * first.translation + alpha * second.translation,
            first.rotation.slerp(alpha, second.rotation).normalized(),
            first.provider_rotation.slerp(alpha, second.provider_rotation)};
    }
};

struct CloudBuilderPose {
    Vec3f translation = Vec3f::Zero();
    Eigen::Quaternionf rotation = Eigen::Quaternionf::Identity();
};

struct Options {
    fs::path trajectory;
    fs::path output;
    // Optional raw-SLAM frontend stream. This mode reuses the calibrated
    // Pandar decoder but deliberately does not require an already estimated
    // trajectory. SurveyorSLAM builds its own surfels after range collation,
    // so this stream stores endpoints and ray stamps, not CloudBuilder normals.
    fs::path slam_scans_output;
    // Exact ROS time transport for the raw-SLAM framing protocol. Legacy
    // cloud-building callers retain the float64-seconds wire format.
    bool frame_timestamps_ns = false;
    fs::path retained_shards;
    fs::path unvoxelized_output;
    fs::path scan_stats;
    float resolution = 0.01F;
    // The post-processing command overrides the sensor-frame limits with the
    // standard indoor profile seen in the reference run.
    float minimum_range = 0.4F;
    float maximum_range = 30.0F;
    // Upper bound for the exact-ray write buffer.  The original surface
    // filter receives every (origin, endpoint) pair, so retained ray history
    // must not average several segments into one representative segment.
    std::size_t max_active_voxels = 1'000'000U;
    int surface_minimum_neighbors = 3;
    float ray_origin_cell = 0.50F;
    std::optional<AxisAlignedRegion> world_region;
    bool shards_only = false;
    bool multilayer_fringe_filter = true;
    bool vertical_foot_filter = true;
    bool no_motion_filter = true;
    bool scan_stats_only = false;
    // Run every CloudBuilder decision through the final rejection mask, but
    // skip point emission and voxel/shard construction. This keeps full-stream
    // alignment checks cheap without changing the normal production path.
    bool filter_stats_only = false;
    PrecisePose rig_from_horiz;
    PrecisePose rig_from_vert;
    PrecisePose vert_box_from_vert;
};

struct Accumulator {
    Vec3f xyz_sum = Vec3f::Zero();
    Vec3f origin_sum = Vec3f::Zero();
    Vec3f normal_sum = Vec3f::Zero();
    float intensity_sum = 0.0F;
    std::uint32_t count = 0U;
};

// Version-2 retained shard record.  Newly written records are exact raw rays
// (count == 1).  Keeping the count field preserves read compatibility with
// older clustered shards, but those older records cannot reproduce the
// original 6 mm segment-to-centroid test exactly.
struct DiskRecordV2 {
    std::int32_t key_x;
    std::int32_t key_y;
    std::int32_t key_z;
    float xyz_x;
    float xyz_y;
    float xyz_z;
    float origin_x;
    float origin_y;
    float origin_z;
    float normal_x;
    float normal_y;
    float normal_z;
    float intensity;
    std::uint32_t count;
};
static_assert(sizeof(DiskRecordV2) == 56U);

#pragma pack(push, 1)
struct PlyPoint {
    float x;
    float y;
    float z;
    float nx;
    float ny;
    float nz;
    std::uint8_t red;
    std::uint8_t green;
    std::uint8_t blue;
    float intensity;
};

struct UnvoxelizedPoint {
    float x;
    float y;
    float z;
    float sensor_x;
    float sensor_y;
    float sensor_z;
    float normal_sensor_x;
    float normal_sensor_y;
    float normal_sensor_z;
    float intensity;
    float origin_x;
    float origin_y;
    float origin_z;
    double packet_timestamp;
    double timestamp;
    std::uint16_t ring;
    std::uint8_t sensor;
    std::uint8_t block;
};
#pragma pack(pop)

static_assert(sizeof(UnvoxelizedPoint) == 72U);

int floorDivision(int value, int denominator) {
    int quotient = value / denominator;
    const int remainder = value % denominator;
    if (remainder != 0 && ((remainder < 0) != (denominator < 0))) {
        --quotient;
    }
    return quotient;
}

double rosSecondsFromNanoseconds(std::int64_t timestamp_ns) {
    constexpr std::int64_t kNanosecondsPerSecond = 1'000'000'000LL;
    const std::int64_t seconds = timestamp_ns / kNanosecondsPerSecond;
    const std::int64_t nanoseconds = timestamp_ns % kNanosecondsPerSecond;
    // ros::Time::toSec() converts sec and nsec separately. Converting the
    // combined epoch nanoseconds first loses low bits before multiplication
    // and changes a few packet-relative float timestamps by one ULP.
    return static_cast<double>(seconds) +
        static_cast<double>(nanoseconds) * 1.0e-9;
}

std::vector<float> parseFloats(const std::string& text) {
    std::vector<float> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::stof(token));
    }
    return values;
}

std::vector<double> parseDoubles(const std::string& text) {
    std::vector<double> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        values.push_back(std::stod(token));
    }
    return values;
}

PrecisePose parsePose(const std::string& text) {
    const auto values = parseDoubles(text);
    if (values.size() != 7U) {
        throw std::invalid_argument("pose must be tx,ty,tz,qx,qy,qz,qw");
    }
    PrecisePose pose;
    pose.translation = Eigen::Vector3d(values[0], values[1], values[2]);
    pose.rotation = Eigen::Quaterniond(
        values[6], values[3], values[4], values[5]).normalized();
    pose.provider_rotation = pose.rotation;
    return pose;
}

AxisAlignedRegion parseRegion(const std::string& text) {
    const auto values = parseFloats(text);
    if (values.size() != 6U) {
        throw std::invalid_argument("world ROI must be minx,miny,minz,maxx,maxy,maxz");
    }
    AxisAlignedRegion region;
    region.minimum = Vec3f(values[0], values[1], values[2]);
    region.maximum = Vec3f(values[3], values[4], values[5]);
    if ((region.maximum.array() <= region.minimum.array()).any()) {
        throw std::invalid_argument("world ROI maximum must exceed minimum on every axis");
    }
    return region;
}

Options parseArguments(int argc, char** argv) {
    Options options;
    for (int i = 1; i < argc; ++i) {
        const std::string argument = argv[i];
        const auto value = [&]() -> std::string {
            if (++i >= argc) {
                throw std::invalid_argument("missing value after " + argument);
            }
            return argv[i];
        };
        if (argument == "--trajectory") {
            options.trajectory = value();
        } else if (argument == "--output") {
            options.output = value();
        } else if (argument == "--slam-scans-output") {
            options.slam_scans_output = value();
        } else if (argument == "--frame-timestamps-ns") {
            options.frame_timestamps_ns = true;
        } else if (argument == "--resolution") {
            options.resolution = std::stof(value());
        } else if (argument == "--min-range") {
            options.minimum_range = std::stof(value());
        } else if (argument == "--max-range") {
            options.maximum_range = std::stof(value());
        } else if (argument == "--max-active-voxels") {
            options.max_active_voxels = std::stoull(value());
        } else if (argument == "--surface-min-neighbors") {
            options.surface_minimum_neighbors = std::stoi(value());
        } else if (argument == "--ray-origin-cell") {
            options.ray_origin_cell = std::stof(value());
        } else if (argument == "--world-roi") {
            options.world_region = parseRegion(value());
        } else if (argument == "--retain-shards") {
            options.retained_shards = value();
        } else if (argument == "--unvoxelized-output") {
            options.unvoxelized_output = value();
        } else if (argument == "--scan-stats") {
            options.scan_stats = value();
        } else if (argument == "--scan-stats-only") {
            options.scan_stats_only = true;
        } else if (argument == "--filter-stats-only") {
            options.filter_stats_only = true;
        } else if (argument == "--shards-only") {
            options.shards_only = true;
        } else if (argument == "--no-multilayer-fringe") {
            options.multilayer_fringe_filter = false;
        } else if (argument == "--no-vertical-foot-filter") {
            options.vertical_foot_filter = false;
        } else if (argument == "--no-motion-filter") {
            options.no_motion_filter = false;
        } else if (argument == "--horiz-pose") {
            options.rig_from_horiz = parsePose(value());
        } else if (argument == "--vert-pose") {
            options.rig_from_vert = parsePose(value());
        } else if (argument == "--vert-box-pose") {
            options.vert_box_from_vert = parsePose(value());
        } else if (argument == "--help") {
            std::cout << "Usage: navvis_recon_pandar --trajectory trajectory.csv --output cloud.ply "
                         "--resolution 0.01 --horiz-pose tx,ty,tz,qx,qy,qz,qw "
                         "--vert-pose tx,ty,tz,qx,qy,qz,qw "
                         "--vert-box-pose tx,ty,tz,qx,qy,qz,qw "
                         "[--surface-min-neighbors 3] [--retain-shards DIR] "
                         "[--unvoxelized-output cloud-before-voxel.ply] "
                         "[--scan-stats scans.csv] [--scan-stats-only] "
                         "[--filter-stats-only] "
                         "[--no-multilayer-fringe] "
                         "[--no-vertical-foot-filter] "
                         "[--no-motion-filter] "
                         "[--ray-origin-cell 0.50] "
                         "[--world-roi minx,miny,minz,maxx,maxy,maxz] [--shards-only]\n"
                         "       navvis_recon_pandar --slam-scans-output scans.bin "
                         "--min-range 0.5 --max-range 60 "
                         "--horiz-pose ... --vert-pose ... [--frame-timestamps-ns]\n"
                         "Reads framed Pandar XTM or Velodyne VLP16 packets from stdin: "
                         "uint8 sensor, float64 time (or int64 ns), uint16 size, payload. "
                         "A zero-size "
                         "frame marks the beginning of a complete laser scan.\n";
            std::exit(0);
        } else {
            throw std::invalid_argument("unknown argument: " + argument);
        }
    }
    if (options.slam_scans_output.empty() &&
        (options.trajectory.empty() || options.output.empty())) {
        throw std::invalid_argument("--trajectory and --output are required");
    }
    if (!options.slam_scans_output.empty() &&
        (!options.trajectory.empty() || !options.output.empty())) {
        throw std::invalid_argument(
            "--slam-scans-output is a standalone raw frontend mode");
    }
    if (!(options.resolution > 0.0F) || !(options.ray_origin_cell > 0.0F) ||
        options.surface_minimum_neighbors < 0) {
        throw std::invalid_argument("resolution and surface filter options are invalid");
    }
    if (options.shards_only && options.retained_shards.empty()) {
        throw std::invalid_argument("--shards-only requires --retain-shards DIR");
    }
    if (options.scan_stats_only && options.filter_stats_only) {
        throw std::invalid_argument(
            "--scan-stats-only and --filter-stats-only are mutually exclusive");
    }
    return options;
}

std::vector<PrecisePose> readTrajectory(const fs::path& path) {
    std::ifstream input(path);
    if (!input) {
        throw std::runtime_error("cannot open trajectory: " + path.string());
    }
    std::vector<PrecisePose> result;
    std::string line;
    while (std::getline(input, line)) {
        if (line.empty() || line[0] == '#') {
            continue;
        }
        std::replace(line.begin(), line.end(), ',', ' ');
        std::stringstream stream(line);
        PrecisePose pose;
        std::string timestamp_text;
        double qx = 0.0;
        double qy = 0.0;
        double qz = 0.0;
        double qw = 1.0;
        if (stream >> timestamp_text >> pose.translation.x() >> pose.translation.y() >>
                pose.translation.z() >> qx >> qy >> qz >> qw) {
            pose.timestamp = std::stod(timestamp_text);
            pose.timestamp_us = static_cast<std::int64_t>(
                std::stold(timestamp_text) * 1.0e6L);
            pose.provider_rotation = Eigen::Quaterniond(qw, qx, qy, qz);
            pose.rotation = pose.provider_rotation.normalized();
            result.push_back(pose);
        }
    }
    std::sort(result.begin(), result.end(),
              [](const PrecisePose& first, const PrecisePose& second) {
                  return first.timestamp < second.timestamp;
              });
    if (result.size() < 2U) {
        throw std::runtime_error("trajectory has fewer than two poses");
    }
    return result;
}

CloudBuilderPose cloudBuilderSupportPose(const PrecisePose& pose) {
    return CloudBuilderPose{
        pose.translation.cast<float>(), pose.provider_rotation.cast<float>()};
}

CloudBuilderPose poseAtCloudBuilder(
    const std::vector<PrecisePose>& trajectory, std::int64_t query_us) {
    const auto upper = std::upper_bound(
        trajectory.begin(), trajectory.end(), query_us,
        [](std::int64_t time_us, const PrecisePose& pose) {
            return time_us < pose.timestamp_us;
        });
    if (upper == trajectory.begin()) {
        return cloudBuilderSupportPose(trajectory.front());
    }
    if (upper == trajectory.end()) {
        return cloudBuilderSupportPose(trajectory.back());
    }

    const PrecisePose& first = *std::prev(upper);
    const PrecisePose& second = *upper;
    const double numerator = static_cast<double>(query_us - first.timestamp_us) * 1.0e-6;
    const double denominator =
        static_cast<double>(second.timestamp_us - first.timestamp_us) * 1.0e-6;
    const double alpha = numerator / denominator;

    // The interpolation helper works in double, but the provider's
    // Rigid3<float> support poses have already been promoted from float.  Match
    // that storage round trip before blending and cast the result back once.
    const Eigen::Vector3d first_translation =
        first.translation.cast<float>().cast<double>();
    const Eigen::Vector3d second_translation =
        second.translation.cast<float>().cast<double>();
    const Eigen::Vector3d translation =
        first_translation + (second_translation - first_translation) * alpha;

    const Eigen::Quaterniond first_rotation(
        first.provider_rotation.cast<float>().cast<double>());
    const Eigen::Quaterniond second_rotation(
        second.provider_rotation.cast<float>().cast<double>());
    const double dot = first_rotation.coeffs().dot(second_rotation.coeffs());
    Eigen::Quaterniond rotation;
    if (std::abs(dot) >= 1.0 - std::numeric_limits<double>::epsilon()) {
        rotation.coeffs() = (1.0 - alpha) * first_rotation.coeffs() +
            (dot < 0.0 ? -alpha : alpha) * second_rotation.coeffs();
    } else {
        const double angle = std::acos(std::abs(dot));
        const double denominator_sine = std::sin(angle);
        const double first_weight =
            std::sin((1.0 - alpha) * angle) / denominator_sine;
        const double second_weight = std::sin(alpha * angle) / denominator_sine *
            (dot < 0.0 ? -1.0 : 1.0);
        rotation.coeffs() = first_weight * first_rotation.coeffs() +
            second_weight * second_rotation.coeffs();
    }
    return CloudBuilderPose{translation.cast<float>(), rotation.cast<float>()};
}

CloudBuilderPose composeCloudBuilderPose(
    const CloudBuilderPose& world_from_rig, const PrecisePose& rig_from_sensor) {
    const Eigen::Quaternionf rig_rotation = rig_from_sensor.rotation.cast<float>();
    const Vec3f rig_translation = rig_from_sensor.translation.cast<float>();
    return CloudBuilderPose{
        world_from_rig.rotation * rig_translation + world_from_rig.translation,
        world_from_rig.rotation * rig_rotation};
}

Eigen::Matrix4f cloudBuilderPoseMatrix(const CloudBuilderPose& pose) {
    Eigen::Matrix4f matrix = Eigen::Matrix4f::Identity();
    matrix.topLeftCorner<3, 3>() = pose.rotation.toRotationMatrix();
    matrix.topRightCorner<3, 1>() = pose.translation;
    return matrix;
}

CloudBuilderPose inverseCloudBuilderPose(const CloudBuilderPose& pose) {
    const Eigen::Quaternionf inverse_rotation = pose.rotation.inverse();
    return CloudBuilderPose{
        -(inverse_rotation * pose.translation), inverse_rotation};
}

Vec3f applyCloudBuilderMatrix(
    const Eigen::Matrix4f& matrix, const Vec3f& point) {
    Eigen::Vector4f transformed = point.z() * matrix.col(2) + matrix.col(3);
    transformed += point.y() * matrix.col(1);
    transformed += point.x() * matrix.col(0);
    return transformed.head<3>();
}

std::uint64_t tilecloudMortonKey(const Vec3f& point) {
    // Tilecloud names encode the low 20 two's-complement bits of the 5 m
    // tile coordinate in x/y/z Morton order.  Surface reads those files in
    // ascending name order, so CompactOctree's lattice is anchored by the
    // first point in the lowest occupied Morton tile rather than by the first
    // chronological return.
    constexpr float tile_size = 5.0F;
    constexpr std::uint32_t coordinate_mask = (1U << 20U) - 1U;
    const std::array<std::int32_t, 3> coordinate{
        static_cast<std::int32_t>(std::floor(point.x() / tile_size)),
        static_cast<std::int32_t>(std::floor(point.y() / tile_size)),
        static_cast<std::int32_t>(std::floor(point.z() / tile_size))};
    std::uint64_t key = 0U;
    for (std::uint32_t bit = 0U; bit < 20U; ++bit) {
        for (std::uint32_t axis = 0U; axis < 3U; ++axis) {
            const std::uint32_t value = static_cast<std::uint32_t>(coordinate[axis]) &
                                        coordinate_mask;
            key |= static_cast<std::uint64_t>((value >> bit) & 1U) << (3U * bit + axis);
        }
    }
    return key;
}

PrecisePose poseAt(const std::vector<PrecisePose>& trajectory, double timestamp) {
    const auto upper = std::upper_bound(
        trajectory.begin(), trajectory.end(), timestamp,
        [](double time, const PrecisePose& pose) { return time < pose.timestamp; });
    if (upper == trajectory.begin()) {
        return trajectory.front();
    }
    if (upper == trajectory.end()) {
        return trajectory.back();
    }
    return PrecisePose::interpolate(*std::prev(upper), *upper, timestamp);
}

std::uint16_t littleU16(const std::vector<std::uint8_t>& data, std::size_t offset) {
    return static_cast<std::uint16_t>(data[offset]) |
           static_cast<std::uint16_t>(data[offset + 1]) << 8U;
}

class StreamingVoxelCloud {
public:
    explicit StreamingVoxelCloud(Options options, CloudBuilderTiming& timing)
        : options_(std::move(options)),
          temporary_directory_(options_.output.string() + ".recon_tmp"),
          timing_(timing) {
        fs::create_directories(options_.output.parent_path());
        if (fs::exists(temporary_directory_)) {
            fs::remove_all(temporary_directory_);
        }
        fs::create_directories(temporary_directory_);
        active_rays_.reserve(options_.max_active_voxels);
    }

    ~StreamingVoxelCloud() {
        if (!finished_ && fs::exists(temporary_directory_)) {
            std::cerr << "Temporary shards retained after failure: " << temporary_directory_ << '\n';
        }
    }

    void add(const Vec3f& xyz, const Vec3f& origin, const Vec3f& normal, float intensity) {
        const VoxelKey endpoint_key{
            static_cast<int>(std::floor(xyz.x() / options_.resolution)),
            static_cast<int>(std::floor(xyz.y() / options_.resolution)),
            static_cast<int>(std::floor(xyz.z() / options_.resolution))};
        const std::uint64_t tilecloud_key = tilecloudMortonKey(xyz);
        if (!has_freespace_anchor_ || tilecloud_key < freespace_anchor_tile_key_) {
            freespace_anchor_ = xyz;
            freespace_anchor_tile_key_ = tilecloud_key;
            has_freespace_anchor_ = true;
        }
        active_rays_.push_back({endpoint_key.x, endpoint_key.y, endpoint_key.z,
                                xyz.x(), xyz.y(), xyz.z(),
                                origin.x(), origin.y(), origin.z(),
                                normal.x(), normal.y(), normal.z(), intensity, 1U});
        if (active_rays_.size() >= options_.max_active_voxels) {
            flush();
        }
    }

    void finish() {
        flush();
        if (!options_.retained_shards.empty()) {
            retainRawShards();
        }
        if (options_.shards_only) {
            fs::remove_all(temporary_directory_);
            finished_ = true;
            std::cerr << "Retained raw shards in " << options_.retained_shards << '\n';
            return;
        }
        const fs::path point_records = temporary_directory_ / "points.bin";
        std::ofstream point_output(point_records, std::ios::binary);
        std::uint64_t point_count = 0U;
        std::uint64_t surface_rejected = 0U;
        for (const auto& entry : fs::directory_iterator(temporary_directory_)) {
            if (entry.path().extension() != ".raytile") {
                continue;
            }
            std::unordered_map<VoxelKey, Accumulator, VoxelKeyHash> tile;
            std::ifstream input(entry.path(), std::ios::binary);
            DiskRecordV2 record{};
            while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
                const VoxelKey key{record.key_x, record.key_y, record.key_z};
                auto& accumulator = tile[key];
                accumulator.xyz_sum += Vec3f(record.xyz_x, record.xyz_y, record.xyz_z);
                accumulator.origin_sum += Vec3f(record.origin_x, record.origin_y, record.origin_z);
                accumulator.normal_sum += Vec3f(record.normal_x, record.normal_y, record.normal_z);
                accumulator.intensity_sum += record.intensity;
                accumulator.count += record.count;
            }
            // Bounded-memory approximation of the recovered density/outlier
            // stages. Count occupied 1 cm voxels in neighboring 10 cm cells,
            // then remove isolated surface returns before writing the tile.
            // The full CloudSurfaceFilter implementation remains available in
            // the library for smaller in-memory clouds.
            const int density_cell_voxels = std::max(
                1, static_cast<int>(std::ceil(0.10F / options_.resolution)));
            std::unordered_map<VoxelKey, std::uint32_t, VoxelKeyHash> density;
            density.reserve(tile.size());
            for (const auto& item : tile) {
                const VoxelKey coarse{floorDivision(item.first.x, density_cell_voxels),
                                      floorDivision(item.first.y, density_cell_voxels),
                                      floorDivision(item.first.z, density_cell_voxels)};
                ++density[coarse];
            }
            for (const auto& item : tile) {
                if (options_.surface_minimum_neighbors > 0) {
                    const VoxelKey coarse{floorDivision(item.first.x, density_cell_voxels),
                                          floorDivision(item.first.y, density_cell_voxels),
                                          floorDivision(item.first.z, density_cell_voxels)};
                    std::uint32_t neighbors = 0U;
                    for (int dz = -1; dz <= 1; ++dz) {
                        for (int dy = -1; dy <= 1; ++dy) {
                            for (int dx = -1; dx <= 1; ++dx) {
                                const auto found = density.find(
                                    VoxelKey{coarse.x + dx, coarse.y + dy, coarse.z + dz});
                                if (found != density.end()) {
                                    neighbors += found->second;
                                }
                            }
                        }
                    }
                    if (neighbors < static_cast<std::uint32_t>(options_.surface_minimum_neighbors)) {
                        ++surface_rejected;
                        continue;
                    }
                }
                const auto& accumulator = item.second;
                if (accumulator.count == 0U) {
                    continue;
                }
                const float inverse_count = 1.0F / static_cast<float>(accumulator.count);
                const Vec3f xyz = inverse_count * accumulator.xyz_sum;
                const Vec3f normal = normalizedOr(accumulator.normal_sum);
                const float intensity = std::clamp(
                    inverse_count * accumulator.intensity_sum, 0.0F, 1.0F);
                const auto gray = static_cast<std::uint8_t>(
                    std::lround(255.0F * intensity));
                const PlyPoint point{xyz.x(), xyz.y(), xyz.z(), normal.x(), normal.y(), normal.z(),
                                     gray, gray, gray, intensity};
                point_output.write(reinterpret_cast<const char*>(&point), sizeof(point));
                ++point_count;
            }
        }
        point_output.close();
        writePly(point_records, point_count);
        fs::remove_all(temporary_directory_);
        finished_ = true;
        std::cerr << "Surface filter removed " << surface_rejected << " isolated voxels\n";
        std::cerr << "Wrote " << point_count << " voxel points to " << options_.output << '\n';
    }

private:
    void retainRawShards() {
        const auto started = SteadyClock::now();
        fs::create_directories(options_.retained_shards);
        for (const auto& existing : fs::directory_iterator(options_.retained_shards)) {
            if (existing.path().extension() == ".tile" ||
                existing.path().extension() == ".raytile") {
                throw std::runtime_error(
                    "retained-shard directory already contains .tile files: " +
                    options_.retained_shards.string());
            }
        }
        std::size_t count = 0U;
        for (const auto& entry : fs::directory_iterator(temporary_directory_)) {
            if (entry.path().extension() != ".raytile") continue;
            const fs::path destination = options_.retained_shards / entry.path().filename();
            std::error_code error;
            fs::create_hard_link(entry.path(), destination, error);
            if (error) {
                fs::copy_file(entry.path(), destination, fs::copy_options::none);
            }
            ++count;
        }
        if (has_freespace_anchor_) {
            std::ofstream anchor(options_.retained_shards / "freespace_anchor.txt");
            anchor.precision(9);
            anchor << freespace_anchor_.x() << ' ' << freespace_anchor_.y() << ' '
                   << freespace_anchor_.z() << '\n';
        }
        std::cerr << "Retained " << count << " raw shard files in "
                  << options_.retained_shards << '\n';
        timing_.retain += elapsedSeconds(started);
    }

    void flush() {
        if (active_rays_.empty()) {
            return;
        }
        const auto partition_started = SteadyClock::now();
        std::unordered_map<VoxelKey, std::vector<DiskRecordV2>, VoxelKeyHash> by_tile;
        by_tile.reserve(128U);
        const int tile_voxels = std::max(
            1, static_cast<int>(std::llround(10.0 / options_.resolution)));  // Always 10 m tiles.
        for (const auto& record : active_rays_) {
            const VoxelKey tile{floorDivision(record.key_x, tile_voxels),
                                floorDivision(record.key_y, tile_voxels),
                                floorDivision(record.key_z, tile_voxels)};
            by_tile[tile].push_back(record);
        }
        timing_.shard_partition += elapsedSeconds(partition_started);
        const auto write_started = SteadyClock::now();
        for (const auto& tile : by_tile) {
            const VoxelKey& key = tile.first;
            const std::string name = "tile_" + std::to_string(key.x) + "_" +
                                     std::to_string(key.y) + "_" + std::to_string(key.z) +
                                     ".raytile";
            std::ofstream output(temporary_directory_ / name, std::ios::binary | std::ios::app);
            output.write(reinterpret_cast<const char*>(tile.second.data()),
                         static_cast<std::streamsize>(tile.second.size() * sizeof(DiskRecordV2)));
        }
        timing_.shard_write += elapsedSeconds(write_started);
        active_rays_.clear();
    }

    void writePly(const fs::path& records, std::uint64_t point_count) const {
        std::ofstream output(options_.output, std::ios::binary);
        output << "ply\nformat binary_little_endian 1.0\n"
               << "comment navvis_recon streaming Pandar XTM reconstruction\n"
               << "element vertex " << point_count << "\n"
               << "property float x\nproperty float y\nproperty float z\n"
               << "property float nx\nproperty float ny\nproperty float nz\n"
               << "property uchar red\nproperty uchar green\nproperty uchar blue\n"
               << "property float intensity\nend_header\n";
        std::ifstream input(records, std::ios::binary);
        output << input.rdbuf();
    }

    Options options_;
    fs::path temporary_directory_;
    std::vector<DiskRecordV2> active_rays_;
    CloudBuilderTiming& timing_;
    Vec3f freespace_anchor_ = Vec3f::Zero();
    std::uint64_t freespace_anchor_tile_key_ = std::numeric_limits<std::uint64_t>::max();
    bool has_freespace_anchor_ = false;
    bool finished_ = false;
};

class UnvoxelizedCloudWriter {
public:
    explicit UnvoxelizedCloudWriter(const fs::path& output)
        : output_(output), records_(output.string() + ".records") {
        if (output_.empty()) {
            return;
        }
        fs::create_directories(output_.parent_path());
        stream_.open(records_, std::ios::binary | std::ios::trunc);
        if (!stream_) {
            throw std::runtime_error("cannot open unvoxelized point records: " + records_.string());
        }
    }

    ~UnvoxelizedCloudWriter() {
        if (!finished_ && !records_.empty()) {
            std::cerr << "Unvoxelized point records retained after failure: " << records_ << '\n';
        }
    }

    void add(const Vec3f& xyz, const Vec3f& sensor_xyz,
             const Vec3f& normal_sensor, const Vec3f& origin,
             float intensity, double packet_timestamp, double timestamp,
             std::uint16_t ring, std::uint8_t sensor, std::uint8_t block) {
        if (output_.empty()) {
            return;
        }
        const UnvoxelizedPoint point{
            xyz.x(), xyz.y(), xyz.z(),
            sensor_xyz.x(), sensor_xyz.y(), sensor_xyz.z(),
            normal_sensor.x(), normal_sensor.y(), normal_sensor.z(), intensity,
            origin.x(), origin.y(), origin.z(), packet_timestamp, timestamp,
            ring, sensor, block};
        stream_.write(reinterpret_cast<const char*>(&point), sizeof(point));
        ++point_count_;
    }

    void finish() {
        if (output_.empty()) {
            finished_ = true;
            return;
        }
        stream_.close();
        std::ofstream output(output_, std::ios::binary | std::ios::trunc);
        if (!output) {
            throw std::runtime_error("cannot write unvoxelized cloud: " + output_.string());
        }
        output << "ply\nformat binary_little_endian 1.0\n"
               << "comment navvis_recon filtered returns before endpoint voxelization\n"
               << "element vertex " << point_count_ << "\n"
               << "property float x\nproperty float y\nproperty float z\n"
               << "property float sensor_x\nproperty float sensor_y\nproperty float sensor_z\n"
               << "property float normal_sensor_x\nproperty float normal_sensor_y\n"
               << "property float normal_sensor_z\n"
               << "property float intensity\n"
               << "property float origin_x\nproperty float origin_y\nproperty float origin_z\n"
               << "property double packet_timestamp\nproperty double timestamp\nproperty ushort ring\n"
               << "property uchar sensor\nproperty uchar block\nend_header\n";
        std::ifstream records(records_, std::ios::binary);
        output << records.rdbuf();
        output.close();
        records.close();
        fs::remove(records_);
        finished_ = true;
        std::cerr << "Wrote " << point_count_ << " filtered returns before voxelization to "
                  << output_ << '\n';
    }

private:
    fs::path output_;
    fs::path records_;
    std::ofstream stream_;
    std::uint64_t point_count_ = 0U;
    bool finished_ = false;
};

struct DecodedPoint {
    Vec3f xyz = Vec3f::Zero();
    // SDK Cartesian point before NavVis AIC. The installed SLAM filter chain
    // evaluates all Region predicates in this native laser frame.
    Vec3f filter_xyz = Vec3f::Zero();
    Vec3f origin = Vec3f::Zero();
    float intensity = 0.0F;
    float raw_intensity = 0.0F;
    double time_offset = 0.0;
    std::uint16_t ring = 0U;
    std::uint8_t block = 0U;
    bool valid = false;
};

struct BufferedPandarPoint {
    DecodedPoint point;
    double packet_timestamp = 0.0;
    std::int64_t packet_timestamp_ns = 0;
};

struct BufferedPandarPacket {
    std::vector<std::uint8_t> data;
    double timestamp = 0.0;
    std::int64_t timestamp_ns = 0;
};

struct BufferedPandarScan {
    std::vector<BufferedPandarPoint> points;
    std::vector<BufferedPandarPacket> packets;
    std::uint8_t sensor = 0U;
    double header_timestamp = 0.0;
    std::int64_t header_timestamp_ns = 0;
    // The high bit of a zero-size frame marker carries world_builder's
    // add_scans state. Disabled scans still advance the no-motion filter but
    // must not reach point decoding output.
    bool insertion_enabled = true;
    bool active = false;
};

std::array<std::array<DecodedPoint, kRings>, kBlocks> decodePandarPacket(
    const std::vector<std::uint8_t>& data, std::uint8_t sensor, const Options& options) {
    std::array<std::array<DecodedPoint, kRings>, kBlocks> result{};
    if (data.size() != kPacketBytes || data[0] != 0xeeU || data[1] != 0xffU) {
        return result;
    }
    const double distance_unit = 0.001 * static_cast<double>(data[9]);
    const std::uint8_t echo = data[802U];
    const auto& calibration = sensor == 0U ? kHorizontalIntrinsic : kVerticalIntrinsic;
    constexpr std::array<float, kBlocks> single_offsets{{
        -244.368F, -194.368F, -144.368F, -94.368F, -44.368F, 5.632F}};
    constexpr std::array<float, kBlocks> dual_offsets{{
        -94.368F, -94.368F, -44.368F, -44.368F, 5.632F, 5.632F}};
    constexpr std::array<float, kBlocks> triple_offsets{{
        -44.368F, -44.368F, -44.368F, 5.632F, 5.632F, 5.632F}};
    const auto& block_offsets = echo == 0x3dU ? triple_offsets :
        ((echo == 0x39U || echo == 0x3bU || echo == 0x3cU) ? dual_offsets : single_offsets);
    for (int block = 0; block < kBlocks; ++block) {
        const std::size_t block_offset = kBodyOffset + static_cast<std::size_t>(block * kBlockBytes);
        const int packet_azimuth_hundredths =
            static_cast<int>(littleU16(data, block_offset));
        int fired_azimuth_hundredths = packet_azimuth_hundredths % kPandarAngleCount;
        if (fired_azimuth_hundredths < 0) {
            fired_azimuth_hundredths += kPandarAngleCount;
        }
        const auto& sdk_trigonometry = pandarSdkTrigonometry();
        const float sdk_sin_azimuth =
            sdk_trigonometry.azimuth_sin[fired_azimuth_hundredths];
        const float sdk_cos_azimuth =
            sdk_trigonometry.azimuth_cos[fired_azimuth_hundredths];
        for (int ring = 0; ring < kRings; ++ring) {
            const std::size_t point_offset = block_offset + 2U + static_cast<std::size_t>(ring * 4);
            const double distance =
                distance_unit * static_cast<double>(littleU16(data, point_offset));
            // Decode and calibrate first. cloud_builder applies its generic
            // range region to the AIC-corrected endpoint, not to this raw
            // measured distance.
            // The installed NavVis PandarGeneral build uses the block azimuth
            // directly for XTM Cartesian decoding.  Unlike the upstream SDK,
            // it does not add spin_speed*firing_offset to this angle; the
            // per-beam firing table is used only for the point timestamp.
            // cloud_builder converts the SDK point back to spherical form,
            // then applies AIC in this exact order: excentricity, azimuth,
            // cone, range gain/bias, ray parallax, translation.  The extracted
            // gain is one and all translation vectors are zero for both G11
            // scanners, so only the non-zero terms appear here.
            // The installed PandarGeneral decoder runs with coordinate
            // correction enabled.  Its XTM path first shortens the measured
            // range by H*cos(nominal elevation), H=30.5 mm.  Because XTM's
            // per-ring horizontal offsets are all zero, the B term vanishes;
            // transformPoint is also disabled in the runtime object.
            const float sdk_elevation_sin = sdk_trigonometry.elevation_sin[ring];
            const float sdk_elevation_cos = sdk_trigonometry.elevation_cos[ring];
            const double coordinate_correction = static_cast<double>(
                kXtmCoordinateCorrectionH * sdk_elevation_cos);
            const float sdk_distance = static_cast<float>(distance - coordinate_correction);
            const float sdk_horizontal_distance = sdk_elevation_cos * sdk_distance;
            const float sdk_x = sdk_horizontal_distance * sdk_sin_azimuth;
            const float sdk_y = sdk_horizontal_distance * sdk_cos_azimuth;
            const float sdk_z = sdk_distance * sdk_elevation_sin;
            result[block][ring].filter_xyz = Vec3f(sdk_x, sdk_y, sdk_z);

            // surveyorslam_processing_node::convertPointCloudWithAic
            // (image address 0x393ad0) does not reuse the packet angle or the
            // nominal cone angle.  It first converts PandarGeneral's float
            // Cartesian point back to spherical coordinates, applies AIC,
            // and converts it to Cartesian again.  Keep the scalar float
            // instruction order below: changing this to Eigen::norm(),
            // hypot(), double intermediates, or direct packet angles creates
            // a measurable sub-micrometre bias against the installed binary.
            const float sdk_horizontal_squared = sdk_x * sdk_x + sdk_y * sdk_y;
            const float sdk_horizontal = ::sqrtf(sdk_horizontal_squared);
            float corrected_distance =
                ::sqrtf(sdk_z * sdk_z + sdk_horizontal_squared);
            float elevation = ::atan2f(sdk_z, sdk_horizontal);
            float azimuth = ::atan2f(sdk_y, sdk_x);

            // Device AIC order recovered from the same function and the
            // LaserModelMultilayer<float> implementation: excentricity,
            // per-ring azimuth, cone, range gain/bias, range-bias spline,
            // spherical reconstruction, ray parallax, translation.
            azimuth -= calibration.excentricity_amplitude *
                       ::sinf(azimuth + calibration.excentricity_phase);
            azimuth += calibration.azimuth_correction[ring];
            elevation += calibration.cone_correction[ring];
            corrected_distance =
                corrected_distance * 1.0F + calibration.range_bias[ring];
            const auto& range_splines = sensor == 0U ?
                kHorizontalRangeSplineValues : kVerticalRangeSplineValues;
            corrected_distance =
                applyRangeBiasSpline(corrected_distance, range_splines[ring]);

            float sin_azimuth = 0.0F;
            float cos_azimuth = 0.0F;
            float sin_elevation = 0.0F;
            float cos_elevation = 0.0F;
            ::sincosf(azimuth, &sin_azimuth, &cos_azimuth);
            ::sincosf(elevation, &sin_elevation, &cos_elevation);
            result[block][ring].xyz = Vec3f(
                corrected_distance * cos_azimuth * cos_elevation +
                    calibration.ray_parallax_amplitude *
                        ::cosf(azimuth + calibration.ray_parallax_phase),
                corrected_distance * sin_azimuth * cos_elevation +
                    calibration.ray_parallax_amplitude *
                        ::sinf(azimuth + calibration.ray_parallax_phase),
                corrected_distance * sin_elevation);
            result[block][ring].origin = Vec3f::Zero();
            // PandarGeneral's raw-distance gate is followed by the generic
            // cloud_builder range predicate on the AIC-corrected endpoint.
            // A measured return exactly at 30 m can move a few millimetres
            // outside the configured sphere after calibration and must then
            // be discarded.
            if (options.slam_scans_output.empty()) {
                const float corrected_range_squared =
                    result[block][ring].xyz.squaredNorm();
                if (corrected_range_squared <
                        options.minimum_range * options.minimum_range ||
                    corrected_range_squared >
                        options.maximum_range * options.maximum_range) {
                    continue;
                }
            }
            const float raw_intensity = static_cast<float>(data[point_offset + 2U]);

            // G11 optical keep-out / trolley reflection masks used by the
            // standard profile. Low-intensity returns behind the horizontal
            // sensor's lower layers are reflections from the device. Two
            // strut sectors are always suppressed. The vertical sensor has
            // separate body and head reflection regions with their recovered
            // raw-intensity thresholds.
            if (options.slam_scans_output.empty() && sensor == 0U) {
                // Runtime RTTI identifies this as RegionBoolean<float, 3>
                // containing two OR-ed RegionYawRange<float> predicates.
                // These are the exact float members captured from the
                // installed Build ID ea5e2be5d588da812f332ef2246559db488e435c.
                constexpr float first_center = -0x1.38c354p+1F;
                constexpr float second_center = -0x1.1df46cp+1F;
                constexpr float half_width = 0x1.f46bbap-5F;
                const bool strut =
                    regionYawRangeContains(
                        result[block][ring].xyz, first_center, half_width) ||
                    regionYawRangeContains(
                        result[block][ring].xyz, second_center, half_width);
                const bool lower_layer_reflection = ring >= 22 && raw_intensity < 5.0F;
                if (strut || lower_layer_reflection) {
                    continue;
                }
            } else if (options.slam_scans_output.empty() && raw_intensity < 8.0F) {
                float yaw_degrees = std::atan2(result[block][ring].xyz.x(),
                                               result[block][ring].xyz.y()) * 180.0F / kPi;
                if (yaw_degrees < 0.0F) {
                    yaw_degrees += 360.0F;
                }
                const float point_norm = result[block][ring].xyz.norm();
                const float pitch_degrees = point_norm > 0.0F ?
                    std::asin(std::clamp(result[block][ring].xyz.z() / point_norm,
                                         -1.0F, 1.0F)) * 180.0F / kPi : 0.0F;
                const bool body_region =
                    (yaw_degrees <= 25.0F || yaw_degrees >= 335.0F) &&
                    pitch_degrees >= -29.0F && pitch_degrees <= -13.0F;
                const bool head_region =
                    yaw_degrees >= 160.0F && yaw_degrees <= 200.0F &&
                    pitch_degrees >= -27.0F && pitch_degrees <= -15.0F;
                if ((body_region && raw_intensity < 5.0F) || head_region) {
                    continue;
                }
            }
            result[block][ring].raw_intensity = raw_intensity;
            result[block][ring].intensity = std::min(raw_intensity / 100.0F, 1.0F);
            result[block][ring].ring = static_cast<std::uint16_t>(ring);
            result[block][ring].block = static_cast<std::uint8_t>(block);
            // CloudBuilder opens recorded scans in PandarGeneral "realtime"
            // mode and stamps every return with m_dPktTimestamp. SurveyorSLAM
            // instead feeds the firing time to its RangeDataCollator; retain
            // the recovered block/beam table only in the standalone raw-SLAM
            // stream.
            result[block][ring].time_offset = options.slam_scans_output.empty() ?
                0.0 : 1.0e-6 * static_cast<double>(
                    block_offsets[block] + kLaserTimeOffsetMicroseconds[ring]);
            // NavVis' installed PandarXTM profile rejects zero-intensity
            // returns from 3 m onward for both laser_horiz and laser_vert.
            if (sdk_distance * sdk_distance >= 9.0F && raw_intensity < 1.0F) {
                continue;
            }
            result[block][ring].valid = true;
        }
    }
    return result;
}

constexpr float kFringeAzimuthTolerance = 0.0131946886F;
constexpr float kNormalMaximumNeighborDistance = 1.5F;
constexpr float kFringeMinimumProjection = 0.0210000007F;
constexpr float kFringeMaximumPairProjection = 1.5F;
constexpr float kFringeMaximumRange = 10.0F;
constexpr float kFringeMinimumIncidence = 1.1344640255F;  // 65 degrees.
constexpr float kFringeIncidenceRangeFactor = 0.7F;
constexpr float kFringeNormalTolerance = 0.0872664601F;  // 5 degrees.
constexpr float kFringeOpposedDirection = -0.866025388F;
constexpr float kFringeLooseDirection = 0.258819044F;
constexpr double kNoMotionMaximumTranslation = 0.005;
constexpr double kNoMotionMaximumAngleDegrees = 0.05;

float circularAngleDifference(float first, float second) {
    // The installed ordered-cloud helpers do not call remainderf. They keep
    // the input subtraction in float, wrap its absolute value with double
    // fmod, restore the float sign, and only then convert back to float. The
    // operation order matters for samples that sit one ULP from a gate.
    const float delta = first - second;
    const double wrapped =
        ::fmod(static_cast<double>(::fabsf(delta)) + kPiDouble,
               2.0 * kPiDouble) -
        kPiDouble;
    const int sign = (delta > 0.0F) - (delta < 0.0F);
    return ::fabsf(static_cast<float>(static_cast<double>(sign) * wrapped));
}

struct OrderedPandarScan {
    const std::vector<BufferedPandarPoint>* points = nullptr;
    // Every return expressed in the laser frame at the complete scan's
    // header timestamp. cloud_builder performs ordered-neighborhood work in
    // this motion-compensated frame, while retaining the original decoded
    // coordinates for final pointwise trajectory projection.
    std::vector<Vec3f> coordinates;
    std::array<std::vector<std::size_t>, kRings> rows;
    std::array<std::vector<float>, kRings> angles;
    std::vector<Vec3f> normals;
};

int sameRingNeighbor(
    const OrderedPandarScan& scan, int ring, std::size_t column, int offset) {
    const auto& row = scan.rows[ring];
    if (row.empty()) {
        return -1;
    }
    const int size = static_cast<int>(row.size());
    int candidate = (static_cast<int>(column) + offset) % size;
    if (candidate < 0) {
        candidate += size;
    }
    if (circularAngleDifference(
            scan.angles[ring][static_cast<std::size_t>(candidate)],
            scan.angles[ring][column]) >
        std::abs(static_cast<float>(offset)) * kFringeAzimuthTolerance) {
        return -1;
    }
    return static_cast<int>(row[static_cast<std::size_t>(candidate)]);
}

int adjacentRingColumn(
    const OrderedPandarScan& scan, int ring, std::size_t column, int ring_offset) {
    const int adjacent_ring = ring + ring_offset;
    if (adjacent_ring < 0 || adjacent_ring >= kRings ||
        scan.rows[adjacent_ring].empty()) {
        return -1;
    }
    const float angle = scan.angles[ring][column];
    const auto& adjacent_angles = scan.angles[adjacent_ring];
    const auto iterator = std::lower_bound(adjacent_angles.begin(), adjacent_angles.end(), angle);
    std::size_t candidate = 0U;
    if (iterator == adjacent_angles.end()) {
        candidate = adjacent_angles.size() - 1U;
    } else if (iterator != adjacent_angles.begin()) {
        const std::size_t after =
            static_cast<std::size_t>(iterator - adjacent_angles.begin());
        const std::size_t before = after - 1U;
        const float after_difference = std::abs(adjacent_angles[after] - angle);
        const float before_difference = std::abs(adjacent_angles[before] - angle);
        // order_multilayer_cloud's precomputed above/below maps use an
        // ordinary lower_bound over the sorted yaw row. At the -pi/pi seam
        // they do not wrap to the opposite endpoint. Ties select the element
        // returned by lower_bound (14dc50..14dcd5 in cloud_builder).
        candidate = after_difference <= before_difference ? after : before;
    }
    return static_cast<int>(candidate);
}

int adjacentRingNeighbor(
    const OrderedPandarScan& scan, int ring, std::size_t column, int ring_offset) {
    const int adjacent_column =
        adjacentRingColumn(scan, ring, column, ring_offset);
    if (adjacent_column < 0) {
        return -1;
    }
    const float angle = scan.angles[ring][column];
    if (circularAngleDifference(
            scan.angles[ring + ring_offset]
                [static_cast<std::size_t>(adjacent_column)],
            angle) > kFringeAzimuthTolerance) {
        return -1;
    }
    return static_cast<int>(scan.rows[ring + ring_offset]
        [static_cast<std::size_t>(adjacent_column)]);
}

int normalSameRingNeighbor(
    const OrderedPandarScan& scan, int ring, std::size_t column, int offset) {
    const int candidate = sameRingNeighbor(scan, ring, column, offset);
    if (candidate < 0) {
        return -1;
    }
    const std::size_t index = scan.rows[ring][column];
    if ((scan.coordinates[static_cast<std::size_t>(candidate)] -
         scan.coordinates[index]).norm() >= kNormalMaximumNeighborDistance) {
        return -1;
    }
    return candidate;
}

int normalAdjacentRingNeighbor(
    const OrderedPandarScan& scan, int ring, std::size_t column, int ring_offset) {
    const int candidate = adjacentRingNeighbor(scan, ring, column, ring_offset);
    if (candidate < 0) {
        return -1;
    }
    const std::size_t index = scan.rows[ring][column];
    if ((scan.coordinates[static_cast<std::size_t>(candidate)] -
         scan.coordinates[index]).norm() >= kNormalMaximumNeighborDistance) {
        return -1;
    }
    return candidate;
}

struct SmallIndexGroup {
    std::array<int, 2> values{};
    std::size_t size = 0U;

    void push(int value) {
        values[size++] = value;
    }
};

SmallIndexGroup normalAdjacentRingGroup(
    const OrderedPandarScan& scan, int ring, std::size_t column,
    int ring_offset, int radius) {
    SmallIndexGroup result;
    const int adjacent_ring = ring + ring_offset;
    const int center_column =
        adjacentRingColumn(scan, ring, column, ring_offset);
    if (center_column < 0) {
        return result;
    }

    const auto& adjacent_row = scan.rows[adjacent_ring];
    const auto& adjacent_angles = scan.angles[adjacent_ring];
    const int adjacent_size = static_cast<int>(adjacent_row.size());
    const float center_angle = scan.angles[ring][column];
    const Vec3f& center_point =
        scan.coordinates[scan.rows[ring][column]];
    const int half_radius = radius / 2;
    const int first_offset = -half_radius;
    const int last_offset = (radius + 1) / 2 - 1;
    for (int offset = first_offset; offset <= last_offset; ++offset) {
        int candidate_column = (center_column + offset) % adjacent_size;
        if (candidate_column < 0) {
            candidate_column += adjacent_size;
        }
        // Do not emit the mapped center twice when a very short adjacent row
        // wraps an off-center window position back onto it (1d70bf..1d70ce).
        if (candidate_column == center_column && offset != 0) {
            continue;
        }
        const float angular_limit =
            std::max(1, std::abs(offset)) * kFringeAzimuthTolerance;
        if (circularAngleDifference(
                adjacent_angles[static_cast<std::size_t>(candidate_column)],
                center_angle) > angular_limit) {
            continue;
        }
        const int candidate = static_cast<int>(
            adjacent_row[static_cast<std::size_t>(candidate_column)]);
        if ((scan.coordinates[static_cast<std::size_t>(candidate)] - center_point).norm() >=
            kNormalMaximumNeighborDistance) {
            continue;
        }
        result.push(candidate);
    }
    return result;
}

struct NormalCandidate {
    Vec3f normal = Vec3f::Constant(std::numeric_limits<float>::quiet_NaN());
    std::size_t sample_count = 0U;
    float transverse_extent = 0.0F;
    float planarity = 0.0F;
    bool valid = false;
};

struct OrderedPlaneFit {
    Vec3f normal;
    Vec3f eigenvalues;
};

std::optional<OrderedPlaneFit> fitOrderedPlane(
    const OrderedPandarScan& scan, const int* indices, std::size_t index_count,
    const Vec3f& viewpoint_point) {
    if (index_count < 3U) {
        return std::nullopt;
    }
    Vec3f mean = Vec3f::Zero();
    for (std::size_t i = 0; i < index_count; ++i) {
        mean += scan.coordinates[static_cast<std::size_t>(indices[i])];
    }
    mean /= static_cast<float>(index_count);
    Eigen::Matrix3f covariance = Eigen::Matrix3f::Zero();
    for (std::size_t i = 0; i < index_count; ++i) {
        const Vec3f centered =
            scan.coordinates[static_cast<std::size_t>(indices[i])] - mean;
        covariance.noalias() += centered * centered.transpose();
    }
    covariance /= static_cast<float>(index_count);
    const Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> solver(covariance);
    if (solver.info() != Eigen::Success) {
        return std::nullopt;
    }
    OrderedPlaneFit result;
    result.normal = solver.eigenvectors().col(0);
    result.eigenvalues = solver.eigenvalues();
    Vec3f& normal = result.normal;
    // SelfAdjointEigenSolver already returns unit eigenvectors.  The installed
    // cloud_builder consumes that vector directly; normalizing it a second
    // time perturbs otherwise exact components by one ULP at incidence gates.
    if (!normal.allFinite()) {
        return std::nullopt;
    }
    if (normal.dot(viewpoint_point) > 0.0F) {
        normal = -normal;
    }
    return result;
}

NormalCandidate estimateOrderedNormalCandidate(
    const OrderedPandarScan& scan, int ring, std::size_t column,
    int same_ring_radius) {
    const int current = static_cast<int>(scan.rows[ring][column]);
    const Vec3f& point = scan.coordinates[static_cast<std::size_t>(current)];
    std::array<SmallIndexGroup, 2> same_ring;
    for (int offset = -same_ring_radius; offset < 0; ++offset) {
        const int neighbor = normalSameRingNeighbor(scan, ring, column, offset);
        if (neighbor >= 0) {
            same_ring[0].push(neighbor);
        }
    }
    for (int offset = 1; offset <= same_ring_radius; ++offset) {
        const int neighbor = normalSameRingNeighbor(scan, ring, column, offset);
        if (neighbor >= 0) {
            same_ring[1].push(neighbor);
        }
    }
    // The binary has two independent window arrays: same-ring {1, 2} and
    // adjacent-ring {1, 1}. Thus both normal candidates use exactly the mapped
    // center return from each adjacent row. The second array is read at
    // config+0x18 before the adjacent collector call (1d75db..1d777e).
    // cloud_builder stores the adjacent groups in upper/lower order.  Keep
    // that order in the combined fit as well: the covariance accumulation is
    // single precision, so swapping these two groups changes the sign of the
    // smallest eigenvalue for nearly degenerate four-point neighborhoods.
    std::array<SmallIndexGroup, 2> adjacent_ring{{
        normalAdjacentRingGroup(scan, ring, column, 1, 1),
        normalAdjacentRingGroup(scan, ring, column, -1, 1)}};

    std::array<Vec3f, 4> quadrant_normals;
    std::size_t quadrant_normal_count = 0U;
    // cloud_builder always evaluates all four adjacent/same-ring pairings.
    // An empty group is still a valid pairing: the remaining group plus the
    // current return may contain the three samples needed by the plane fit.
    // The pair builder appends adjacent-ring returns first, then same-ring
    // returns, and finally the current return.  That order is significant for
    // degenerate float PCA fits.
    for (const auto& adjacent_group : adjacent_ring) {
        for (const auto& same_group : same_ring) {
            std::array<int, 5> plane_indices{};
            std::size_t plane_index_count = 0U;
            for (std::size_t i = 0; i < adjacent_group.size; ++i) {
                plane_indices[plane_index_count++] = adjacent_group.values[i];
            }
            for (std::size_t i = 0; i < same_group.size; ++i) {
                plane_indices[plane_index_count++] = same_group.values[i];
            }
            plane_indices[plane_index_count++] = current;
            if (const auto plane = fitOrderedPlane(
                    scan, plane_indices.data(), plane_index_count, point)) {
                quadrant_normals[quadrant_normal_count++] = plane->normal;
            }
        }
    }

    std::array<int, 9> combined_indices{};
    std::size_t combined_index_count = 0U;
    for (const auto& group : same_ring) {
        for (std::size_t i = 0; i < group.size; ++i) {
            combined_indices[combined_index_count++] = group.values[i];
        }
    }
    for (const auto& group : adjacent_ring) {
        for (std::size_t i = 0; i < group.size; ++i) {
            combined_indices[combined_index_count++] = group.values[i];
        }
    }
    combined_indices[combined_index_count++] = current;

    NormalCandidate result;
    result.sample_count = combined_index_count;
    const auto combined_plane = fitOrderedPlane(
        scan, combined_indices.data(), combined_index_count, point);
    if (!combined_plane) {
        return result;
    }
    const float middle_eigenvalue = combined_plane->eigenvalues.y();
    const float smallest_eigenvalue = combined_plane->eigenvalues.x();
    // The binary reports a fixed three-sigma full width for both principal
    // variances.  It computes the square roots in double, converts to float,
    // floors each width at 2*float-epsilon, then takes their ratio.
    const float transverse_extent = static_cast<float>(
        2.0 * std::sqrt(3.0 * static_cast<double>(middle_eigenvalue)));
    const float normal_extent = static_cast<float>(
        2.0 * std::sqrt(3.0 * static_cast<double>(smallest_eigenvalue)));
    constexpr float kMinimumExtent = 2.0F * std::numeric_limits<float>::epsilon();
    result.transverse_extent = transverse_extent;
    result.planarity = 1.0F -
        std::max(normal_extent, kMinimumExtent) /
        std::max(transverse_extent, kMinimumExtent);
    if (quadrant_normal_count >= 2U) {
        Vec3f normal_sum = Vec3f::Zero();
        for (std::size_t i = 0; i < quadrant_normal_count; ++i) {
            normal_sum += quadrant_normals[i];
        }
        const float norm = normal_sum.norm();
        if (normal_sum.allFinite() && std::isfinite(norm) && norm > 1.0e-8F) {
            result.normal = normal_sum / norm;
            result.valid = true;
            return result;
        }
    }
    result.normal = combined_plane->normal;
    result.valid = true;
    return result;
}

OrderedPandarScan orderPandarScan(
    const std::vector<BufferedPandarPoint>& points,
    std::vector<Vec3f> motion_compensated_coordinates) {
    OrderedPandarScan result;
    result.points = &points;
    result.coordinates = std::move(motion_compensated_coordinates);
    result.normals.assign(
        points.size(),
        Vec3f::Constant(std::numeric_limits<float>::quiet_NaN()));
    std::vector<float> point_angles(points.size());
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < points.size(); ++index) {
        const Vec3f& xyz = result.coordinates[index];
        point_angles[index] = std::atan2(xyz.y(), xyz.x());
    }
    // Preserve packet order while building rows; stable_sort below uses the
    // precomputed azimuth as its key.
    for (std::size_t index = 0; index < points.size(); ++index) {
        result.rows[points[index].point.ring].push_back(index);
    }
    // cloud_builder dispatches ordered scan work through its worker pool. Each
    // ring owns its row/angle arrays, so sorting can run concurrently without
    // changing comparison order inside a ring.
#pragma omp parallel for schedule(static)
    for (int ring = 0; ring < kRings; ++ring) {
        auto& row = result.rows[ring];
        std::stable_sort(row.begin(), row.end(), [&](std::size_t first, std::size_t second) {
            return point_angles[first] < point_angles[second];
        });
        auto& angles = result.angles[ring];
        angles.reserve(row.size());
        for (const std::size_t index : row) {
            angles.push_back(point_angles[index]);
        }
    }

    // cloud_builder evaluates radii {1, 2}. Its candidate selector lets the
    // final radius bypass the local score gates (1db4f0..1db531), but the
    // ordered-cloud assembler applies the same sample-count, extent, and
    // planarity gates once more to the selected result (1dc9b0..1dc9de).
    // Therefore a weak radius-two fallback is returned internally yet still
    // becomes an invalid/NaN point normal, matching the record consumed by
    // the fringe predicate.
    // A normal reads only the now-frozen ordered rows and writes one unique
    // output slot. Parallel evaluation is therefore bitwise deterministic.
#pragma omp parallel for schedule(static)
    for (int ring = 0; ring < kRings; ++ring) {
        const auto& row = result.rows[ring];
        for (std::size_t column = 0; column < row.size(); ++column) {
            const std::size_t index = row[column];
            const NormalCandidate first =
                estimateOrderedNormalCandidate(result, ring, column, 1);
            if (first.valid && first.sample_count >= 4U &&
                first.transverse_extent >= 0.015F && first.planarity >= 0.1F) {
                result.normals[index] = first.normal;
                continue;
            }
            const NormalCandidate second =
                estimateOrderedNormalCandidate(result, ring, column, 2);
            if (second.valid && second.sample_count >= 4U &&
                second.transverse_extent >= 0.015F && second.planarity >= 0.1F) {
                result.normals[index] = second.normal;
            }
        }
    }
    return result;
}

struct FringeDescriptor {
    Vec3f direction = Vec3f::Zero();
    float projection = 0.0F;
    bool valid = false;
};

FringeDescriptor fringeDescriptor(
    const Vec3f& point, const Vec3f& ray, const Vec3f& neighbor,
    bool point_minus_neighbor) {
    FringeDescriptor result;
    const Vec3f difference = point_minus_neighbor ? point - neighbor : neighbor - point;
    const float length = difference.norm();
    if (!std::isfinite(length) || length <= 0.0F) {
        return result;
    }
    result.direction = difference / length;
    // The binary computes dot(ray, normalized_difference) * length.
    result.projection = ray.dot(result.direction) * length;
    result.valid = true;
    return result;
}

bool fringeNormalConsistent(
    const OrderedPandarScan& scan, const Vec3f& ray, float incidence,
    int neighbor) {
    if (neighbor < 0) {
        return false;
    }
    const Vec3f& normal = scan.normals[static_cast<std::size_t>(neighbor)];
    if (!normal.allFinite()) {
        return false;
    }
    const float neighbor_incidence = std::acos(std::clamp(
        std::abs(ray.dot(normal)), 0.0F, 1.0F));
    return circularAngleDifference(neighbor_incidence, incidence) <
        kFringeNormalTolerance;
}

bool isMultilayerFringe(
    const OrderedPandarScan& scan, int ring, std::size_t column) {
    const std::size_t index = scan.rows[ring][column];
    const Vec3f& point = scan.coordinates[index];
    const Vec3f& normal = scan.normals[index];
    if (!point.allFinite() || !normal.allFinite()) {
        return false;
    }
    const float range = point.norm();
    if (!std::isfinite(range) || range <= 0.0F || range > kFringeMaximumRange) {
        return false;
    }
    const Vec3f ray = point / range;
    const float incidence = std::acos(std::clamp(
        std::abs(ray.dot(normal)), 0.0F, 1.0F));
    const float incidence_range =
        kFringeIncidenceRangeFactor * ::atanf(1.0F / range);
    const float range_dependent_threshold = static_cast<float>(
        0.5 * kPiDouble - static_cast<double>(incidence_range));
    const float incidence_threshold =
        std::max(kFringeMinimumIncidence, range_dependent_threshold);
    if (!(incidence > incidence_threshold)) {
        return false;
    }

    std::array<int, 3> previous{{
        sameRingNeighbor(scan, ring, column, -3),
        sameRingNeighbor(scan, ring, column, -2),
        sameRingNeighbor(scan, ring, column, -1)}};
    std::array<int, 3> following{{
        sameRingNeighbor(scan, ring, column, 1),
        sameRingNeighbor(scan, ring, column, 2),
        sameRingNeighbor(scan, ring, column, 3)}};
    // The binary's three-neighbor collector keeps a same-ring run contiguous:
    // when the nearest sample fails its angular gate, it explicitly clears the
    // two farther optionals (1244c6..1244fe). It does not allow a gap followed
    // by a farther valid sample.
    if (previous[2] < 0) {
        previous[0] = -1;
        previous[1] = -1;
    }
    if (following[0] < 0) {
        following[1] = -1;
        following[2] = -1;
    }
    const int lower = adjacentRingNeighbor(scan, ring, column, -1);
    const int upper = adjacentRingNeighbor(scan, ring, column, 1);

    const FringeDescriptor a = previous[2] < 0 ? FringeDescriptor{} :
        fringeDescriptor(
            point, ray,
            scan.coordinates[static_cast<std::size_t>(previous[2])], true);
    const FringeDescriptor b = following[0] < 0 ? FringeDescriptor{} :
        fringeDescriptor(
            point, ray,
            scan.coordinates[static_cast<std::size_t>(following[0])], false);
    const FringeDescriptor c = lower < 0 ? FringeDescriptor{} :
        fringeDescriptor(
            point, ray, scan.coordinates[static_cast<std::size_t>(lower)], true);
    const FringeDescriptor d = upper < 0 ? FringeDescriptor{} :
        fringeDescriptor(
            point, ray, scan.coordinates[static_cast<std::size_t>(upper)], false);

    const bool small_ab =
        (a.valid && std::abs(a.projection) < kFringeMinimumProjection) ||
        (b.valid && std::abs(b.projection) < kFringeMinimumProjection);
    const bool small_cd =
        (c.valid && std::abs(c.projection) < kFringeMinimumProjection) ||
        (d.valid && std::abs(d.projection) < kFringeMinimumProjection);
    const float sum_ab =
        (a.valid ? a.projection : 0.0F) + (b.valid ? b.projection : 0.0F);
    const float sum_cd =
        (c.valid ? c.projection : 0.0F) + (d.valid ? d.projection : 0.0F);
    if (std::abs(sum_ab) > kFringeMaximumPairProjection ||
        std::abs(sum_cd) > kFringeMaximumPairProjection ||
        (small_ab && small_cd)) {
        return false;
    }

    const bool ab_valid = a.valid && b.valid;
    const bool cd_valid = c.valid && d.valid;
    const bool previous_consistent = std::all_of(
        previous.begin(), previous.end(), [&](int neighbor) {
            return fringeNormalConsistent(scan, ray, incidence, neighbor);
        });
    const bool following_consistent = std::all_of(
        following.begin(), following.end(), [&](int neighbor) {
            return fringeNormalConsistent(scan, ray, incidence, neighbor);
        });
    const bool adjacent_consistent =
        fringeNormalConsistent(scan, ray, incidence, lower) ||
        fringeNormalConsistent(scan, ray, incidence, upper);
    const bool consistency =
        (previous_consistent || following_consistent) && adjacent_consistent;
    // The reference has a hard rejection guard for an opposed descriptor
    // pair: such geometry is never classified as fringe, independently of
    // the neighboring-normal vote.  The two-pair, both-loose path also keeps
    // the point unconditionally: at 1c4d87 it evaluates the consistency helper
    // but 1c4d97 jumps to the epilogue without copying EAX into the return byte.
    // All remaining non-opposed cases return the inverse consistency vote.
    if (ab_valid && cd_valid) {
        const float dot_ab = a.direction.dot(b.direction);
        const float dot_cd = c.direction.dot(d.direction);
        if (dot_ab < kFringeOpposedDirection ||
            dot_cd < kFringeOpposedDirection) {
            return false;
        }
        if (dot_ab < kFringeLooseDirection && dot_cd < kFringeLooseDirection) {
            return false;
        }
    } else if (ab_valid) {
        if (a.direction.dot(b.direction) < kFringeOpposedDirection) {
            return false;
        }
    } else if (cd_valid) {
        if (c.direction.dot(d.direction) < kFringeOpposedDirection) {
            return false;
        }
    }

    return !consistency;
}

bool validVelodynePacket(const std::vector<std::uint8_t>& data) {
    if (data.size() != kVelodynePacketBytes) {
        return false;
    }
    for (int block = 0; block < kVelodyneBlocks; ++block) {
        const std::size_t offset = static_cast<std::size_t>(block * kVelodyneBlockBytes);
        if (data[offset] != 0xffU || data[offset + 1U] != 0xeeU) {
            return false;
        }
    }
    return true;
}

std::vector<DecodedPoint> decodeVelodynePacket(
    const std::vector<std::uint8_t>& data, std::uint8_t sensor, const Options& options) {
    std::vector<DecodedPoint> result;
    if (!validVelodynePacket(data)) {
        return result;
    }

    const bool dual_return = data[1204U] == 0x39U;
    result.reserve(static_cast<std::size_t>(kVelodyneBlocks * 2 * kVelodyneRings));
    for (int block = 0; block < kVelodyneBlocks; ++block) {
        const std::size_t block_offset = static_cast<std::size_t>(block * kVelodyneBlockBytes);
        const int current_azimuth = static_cast<int>(littleU16(data, block_offset + 2U));
        const int next_block = dual_return ? block + 2 : block + 1;
        int azimuth_delta = 0;
        if (next_block < kVelodyneBlocks) {
            azimuth_delta = static_cast<int>(littleU16(
                data, static_cast<std::size_t>(next_block * kVelodyneBlockBytes + 2))) -
                current_azimuth;
        } else {
            const int previous_block = dual_return ? block - 2 : block - 1;
            azimuth_delta = current_azimuth - static_cast<int>(littleU16(
                data, static_cast<std::size_t>(previous_block * kVelodyneBlockBytes + 2)));
        }
        if (azimuth_delta < 0) {
            azimuth_delta += 36000;
        }
        // A packet boundary or corrupt block must not create a nearly full-turn
        // interpolation. VLP16 at 20 Hz advances about 0.8 degrees per block.
        if (azimuth_delta > 3000) {
            azimuth_delta = 0;
        }

        const int timing_block = dual_return ? block / 2 : block;
        for (int firing = 0; firing < 2; ++firing) {
            for (int ring = 0; ring < kVelodyneRings; ++ring) {
                const int channel = firing * kVelodyneRings + ring;
                const std::size_t point_offset = block_offset + 4U +
                    static_cast<std::size_t>(channel * 3);
                const float distance = kVelodyneDistanceUnit *
                    static_cast<float>(littleU16(data, point_offset));
                if (distance < options.minimum_range || distance > options.maximum_range) {
                    continue;
                }
                const float firing_fraction =
                    (static_cast<float>(firing) * kVelodyneFiringDurationMicroseconds +
                     static_cast<float>(ring) * kVelodyneChannelDurationMicroseconds) /
                    kVelodyneBlockDurationMicroseconds;
                const float azimuth_degrees = 0.01F *
                    (static_cast<float>(current_azimuth) +
                     firing_fraction * static_cast<float>(azimuth_delta));
                const float azimuth = azimuth_degrees * kPi / 180.0F;
                const float elevation = kVlp16ElevationDegrees[ring] * kPi / 180.0F;
                const float radial = distance * std::cos(elevation);
                const float raw_intensity = static_cast<float>(data[point_offset + 2U]);
                const float minimum_intensity = sensor == 0U ? 2.0F : 1.0F;
                if (raw_intensity < minimum_intensity) {
                    continue;
                }

                DecodedPoint point;
                // Match velodyne_pointcloud's laser-frame convention used by
                // the sensor_frame.xml extrinsics: azimuth zero points +Y.
                point.xyz = Vec3f(
                    radial * std::sin(azimuth),
                    radial * std::cos(azimuth),
                    distance * std::sin(elevation));
                // The installed VLP16db calibration has zero horizontal and
                // vertical emitter offsets, so the calibrated ray origin is
                // the laser-frame origin.
                point.origin = Vec3f::Zero();
                point.intensity = std::min(raw_intensity / 100.0F, 1.0F);
                point.ring = static_cast<std::uint16_t>(ring);
                point.block = static_cast<std::uint8_t>(block);
                point.time_offset = 1.0e-6 * static_cast<double>(
                    static_cast<float>(timing_block) * kVelodyneBlockDurationMicroseconds +
                    static_cast<float>(firing) * kVelodyneFiringDurationMicroseconds +
                    static_cast<float>(ring) * kVelodyneChannelDurationMicroseconds);
                point.valid = true;
                result.push_back(point);
            }
        }
    }
    return result;
}

void emitPoint(
    const DecodedPoint& point, const Vec3f& normal_sensor, std::uint8_t sensor,
    double timestamp, const Options& options, const std::vector<PrecisePose>& trajectory,
    StreamingVoxelCloud& output, UnvoxelizedCloudWriter& unvoxelized_output,
    const Eigen::Quaterniond* normal_world_from_sensor = nullptr) {
    const PrecisePose& rig_from_sensor =
        sensor == 0U ? options.rig_from_horiz : options.rig_from_vert;
    const PrecisePose world_from_rig = poseAt(trajectory, timestamp + point.time_offset);
    const Eigen::Vector3d point_rig = rig_from_sensor.apply(point.xyz.cast<double>());
    const Eigen::Vector3d origin_rig = rig_from_sensor.apply(point.origin.cast<double>());
    const Vec3f world = world_from_rig.apply(point_rig).cast<float>();
    if (options.world_region && !options.world_region->contains(world)) {
        return;
    }
    const Vec3f origin_world = world_from_rig.apply(origin_rig).cast<float>();
    const Vec3f normal_world = normal_world_from_sensor != nullptr ?
        ((*normal_world_from_sensor) * normal_sensor.cast<double>()).cast<float>() :
        (world_from_rig.rotation *
            (rig_from_sensor.rotation * normal_sensor.cast<double>())).cast<float>();
    output.add(world, origin_world, normal_world, point.intensity);
    unvoxelized_output.add(world, point.xyz, normal_sensor, origin_world, point.intensity, timestamp,
                           timestamp + point.time_offset, point.ring, sensor, point.block);
}

void emitTransformedPandarPoint(
    const DecodedPoint& point, const Vec3f& normal_sensor, std::uint8_t sensor,
    double packet_timestamp, const Options& options, const Vec3f& world,
    const Vec3f& origin_world, const Eigen::Quaternionf& normal_world_from_sensor,
    StreamingVoxelCloud& output, UnvoxelizedCloudWriter& unvoxelized_output) {
    if (options.world_region && !options.world_region->contains(world)) {
        return;
    }
    const Vec3f normal_world = normal_world_from_sensor * normal_sensor;
    output.add(world, origin_world, normal_world, point.intensity);
    unvoxelized_output.add(
        world, point.xyz, normal_sensor, origin_world, point.intensity,
        packet_timestamp, packet_timestamp + point.time_offset,
        point.ring, sensor, point.block);
}

std::optional<Eigen::Vector4f> fitFootPlaneRansac(
    const std::vector<Vec3f>& world_coordinates,
    const std::vector<std::size_t>& candidate_indices) {
    if (candidate_indices.size() < 3U) {
        return std::nullopt;
    }

    // This mirrors PCL 1.12 RandomSampleConsensus<...> and
    // SampleConsensusModelPlane<...>, as instantiated in cloud_builder:
    // deterministic seed 12345, persistent partial Fisher-Yates shuffling,
    // 1 cm inlier threshold, probability 0.99 and at most 50 iterations.
    boost::mt19937 random_engine(12345U);
    boost::uniform_int<> random_integer(0, INT_MAX);
    std::vector<std::size_t> shuffled(candidate_indices.size());
    std::iota(shuffled.begin(), shuffled.end(), 0U);

    std::size_t best_inlier_count = 0U;
    Eigen::Vector4f best_coefficients = Eigen::Vector4f::Zero();
    double required_iterations = std::numeric_limits<double>::max();
    int iterations = 0;
    unsigned int skipped_count = 0U;
    constexpr unsigned int maximum_skips =
        static_cast<unsigned int>(kFootRansacMaximumIterations * 10);

    while (true) {
        for (std::size_t sample = 0; sample < 3U; ++sample) {
            const std::size_t remaining = shuffled.size() - sample;
            const std::size_t selected = sample +
                static_cast<std::size_t>(random_integer(random_engine)) % remaining;
            std::swap(shuffled[sample], shuffled[selected]);
        }

        const Vec3f& p0 = world_coordinates[candidate_indices[shuffled[0]]];
        const Vec3f& p1 = world_coordinates[candidate_indices[shuffled[1]]];
        const Vec3f& p2 = world_coordinates[candidate_indices[shuffled[2]]];
        const Vec3f p1p0 = p1 - p0;
        const Vec3f p2p0 = p2 - p0;
        const Vec3f ratios = p1p0.array() / p2p0.array();
        if (ratios.x() == ratios.y() && ratios.z() == ratios.y()) {
            if (++skipped_count < maximum_skips) {
                continue;
            }
            break;
        }

        Eigen::Vector4f coefficients;
        coefficients.x() = p1p0.y() * p2p0.z() - p1p0.z() * p2p0.y();
        coefficients.y() = p1p0.z() * p2p0.x() - p1p0.x() * p2p0.z();
        coefficients.z() = p1p0.x() * p2p0.y() - p1p0.y() * p2p0.x();
        coefficients.w() = 0.0F;
        coefficients.normalize();
        coefficients.w() = -coefficients.dot(Eigen::Vector4f(
            p0.x(), p0.y(), p0.z(), 1.0F));

        std::size_t inlier_count = 0U;
        for (const std::size_t index : candidate_indices) {
            const Vec3f& point = world_coordinates[index];
            // PCL's SSE path groups the additions this way.
            const float distance = std::abs(
                (coefficients.x() * point.x() + coefficients.y() * point.y()) +
                (coefficients.z() * point.z() + coefficients.w()));
            if (distance < kFootRansacDistance) {
                ++inlier_count;
            }
        }

        if (inlier_count > best_inlier_count) {
            best_inlier_count = inlier_count;
            best_coefficients = coefficients;
            const double inlier_ratio = static_cast<double>(inlier_count) /
                static_cast<double>(candidate_indices.size());
            double no_outlier_probability = 1.0 -
                std::pow(inlier_ratio, 3.0);
            no_outlier_probability = std::clamp(
                no_outlier_probability,
                std::numeric_limits<double>::epsilon(),
                1.0 - std::numeric_limits<double>::epsilon());
            required_iterations = std::log(1.0 - 0.99) /
                std::log(no_outlier_probability);
        }

        ++iterations;
        if (static_cast<double>(iterations) > required_iterations ||
            iterations > kFootRansacMaximumIterations) {
            break;
        }
    }
    if (best_inlier_count == 0U) {
        return std::nullopt;
    }
    return best_coefficients;
}

enum class PacketFormat { PandarXtm, VelodyneVlp16 };

void appendPandarPacket(
    BufferedPandarScan& scan, std::vector<std::uint8_t> data, double timestamp,
    std::int64_t timestamp_ns) {
    scan.packets.push_back(
        BufferedPandarPacket{std::move(data), timestamp, timestamp_ns});
}

void decodePandarScan(
    BufferedPandarScan& scan, const Options& options, CloudBuilderTiming& timing) {
    if (scan.packets.empty()) {
        return;
    }
    const auto started = SteadyClock::now();
    using DecodedPacket = std::array<std::array<DecodedPoint, kRings>, kBlocks>;
    std::vector<DecodedPacket> decoded(scan.packets.size());
#pragma omp parallel for schedule(static)
    for (std::size_t packet_index = 0; packet_index < scan.packets.size(); ++packet_index) {
        decoded[packet_index] = decodePandarPacket(
            scan.packets[packet_index].data, scan.sensor, options);
    }
    scan.points.reserve(scan.packets.size() * static_cast<std::size_t>(kBlocks * kRings));
    for (std::size_t packet_index = 0; packet_index < scan.packets.size(); ++packet_index) {
        for (int block = 0; block < kBlocks; ++block) {
            for (int ring = 0; ring < kRings; ++ring) {
                if (decoded[packet_index][block][ring].valid) {
                    scan.points.push_back(BufferedPandarPoint{
                        decoded[packet_index][block][ring],
                        scan.packets[packet_index].timestamp,
                        scan.packets[packet_index].timestamp_ns});
                }
            }
        }
    }
    scan.packets.clear();
    timing.decode += elapsedSeconds(started);
}

#pragma pack(push, 1)
struct SlamScanPoint {
    float x;
    float y;
    float z;
    float time_offset;
};
#pragma pack(pop)

static_assert(sizeof(SlamScanPoint) == 16U);

#pragma pack(push, 1)
struct SlamScanPointNs {
    float x;
    float y;
    float z;
    std::int64_t timestamp_ns;
};
#pragma pack(pop)

static_assert(sizeof(SlamScanPointNs) == 20U);

float absoluteWrappedAngleDifference(float angle, float center) {
    return std::abs(std::remainder(angle - center, 2.0F * kPi));
}

bool inAngularRegion(
    const Eigen::Vector3d& point, float yaw_center, float yaw_half_width,
    float pitch_center, float pitch_half_width) {
    const float x = static_cast<float>(point.x());
    const float y = static_cast<float>(point.y());
    const float z = static_cast<float>(point.z());
    const float yaw = ::atan2f(y, x);
    // RegionPitchRange evaluates sqrtf(x*x + y*y) directly. std::hypotf uses
    // a scaled implementation and rounds one ULP higher for the recovered
    // ring-25 boundary point, which reverses the strict half-width decision.
    const float planar_range = ::sqrtf(x * x + y * y);
    const float pitch = ::atan2f(z, planar_range);
    return absoluteWrappedAngleDifference(yaw, yaw_center) < yaw_half_width &&
           absoluteWrappedAngleDifference(pitch, pitch_center) < pitch_half_width;
}

bool rejectedBySurveyorMinimumRangeFilter(
    std::uint8_t sensor, const Eigen::Vector3d& rig_point) {
    // Runtime serialization of the two RegionFilter predicates installed
    // immediately after Pandar conversion in the G11 offline pipeline:
    //
    //   1. a 0.5 m sphere around base_link;
    //   2. a transformed intersection of a radius-0.3 m cylinder and z <= 0.
    //
    // RegionTransformed stores its Eigen matrix column-major. Keep these
    // products explicit (and in float) to match the binary's predicate rather
    // than passing through Eigen's double-precision transform machinery.
    const float x = static_cast<float>(rig_point.x());
    const float y = static_cast<float>(rig_point.y());
    const float z = static_cast<float>(rig_point.z());
    if (x * x + y * y + z * z <= 0.25F) {
        return true;
    }
    // The transformed lower-body keep-out is attached to laser_vert. Four
    // captured horizontal revolutions pass this second predicate unchanged;
    // applying its matrix to horizontal native coordinates creates a false
    // 117--140 point loss per revolution.
    if (sensor == 0U) {
        return false;
    }
    const float transformed_x =
        -0.006795319262892008F * x + 0.11973387002944946F * y +
        0.9927828907966614F * z;
    const float transformed_y =
        -0.9999668598175049F * x + 0.0036559293512254953F * y -
        0.007285381201654673F * z;
    const float transformed_z =
        -0.004501895513385534F * x - 0.9927994608879089F * y +
        0.11970502138137817F * z;
    return transformed_x * transformed_x + transformed_y * transformed_y <= 0.09F &&
           transformed_z <= 0.0F;
}

bool rejectedBySurveyorSelfFilter(
    std::uint8_t sensor, const Eigen::Vector3d& rig_point, float raw_intensity) {
    // Runtime serialization of the installed G11 IntensityRegionFilter chain:
    //   laser_vert foot: yaw  +90 +/-25 deg, pitch -21 +/-8 deg, I < 5
    //   laser_horiz body: all yaw,          pitch -15 +/-7 deg, I < 5
    //   laser_vert head: yaw  -90 +/-20 deg, pitch -21 +/-6 deg, I < 0.08
    // RegionYawRange/RegionPitchRange use strict half-width comparisons and
    // IntensityRegionFilter also compares intensity strictly.
    constexpr float degrees = kPi / 180.0F;
    const float x = static_cast<float>(rig_point.x());
    const float y = static_cast<float>(rig_point.y());
    const float z = static_cast<float>(rig_point.z());
    const float pitch = ::atan2f(z, ::sqrtf(x * x + y * y));
    if (sensor == 0U) {
        return raw_intensity < 5.0F &&
               absoluteWrappedAngleDifference(pitch, -15.0F * degrees) <
                   7.0F * degrees;
    }
    const bool foot = raw_intensity < 5.0F && inAngularRegion(
        rig_point, 90.0F * degrees, 25.0F * degrees,
        -21.0F * degrees, 8.0F * degrees);
    const bool head = raw_intensity < 0.08F && inAngularRegion(
        rig_point, -90.0F * degrees, 20.0F * degrees,
        -21.0F * degrees, 6.0F * degrees);
    return foot || head;
}

std::size_t flushPandarSlamScan(
    BufferedPandarScan& scan, const Options& options, std::ofstream& output,
    std::uint64_t& fringe_rejected, CloudBuilderTiming& timing) {
    if (!scan.active) {
        return 0U;
    }
    const std::size_t raw_ray_count_value =
        scan.packets.size() * static_cast<std::size_t>(kBlocks * kRings);
    const std::int64_t collator_base_ns =
        (scan.header_timestamp_ns / 1000) * 1000;
    std::vector<std::int64_t> packet_timestamps_ns;
    packet_timestamps_ns.reserve(scan.packets.size());
    for (const BufferedPandarPacket& packet : scan.packets) {
        packet_timestamps_ns.push_back(
            collator_base_ns + static_cast<std::int64_t>(
                (packet.timestamp - scan.header_timestamp) * 1.0e9));
    }
    decodePandarScan(scan, options, timing);
    const std::size_t input_count = scan.points.size();
    if (scan.points.empty()) {
        scan = BufferedPandarScan{};
        return input_count;
    }

    // The installed SurveyorSLAM RangeDataCollator receives calibrated ray
    // endpoints directly. It keeps invalid rays for the 58,000-ray trigger,
    // then estimates scan surfels after a batch is complete. In this compact
    // archive the header preserves the raw slot count while the point array
    // stores only finite endpoints used by geometry.
    const PrecisePose& rig_from_sensor =
        scan.sensor == 0U ? options.rig_from_horiz : options.rig_from_vert;
    std::vector<SlamScanPointNs> accepted;
    accepted.reserve(scan.points.size());
    for (std::size_t index = 0; index < scan.points.size(); ++index) {
        const Eigen::Vector3d filter_point =
            scan.points[index].point.filter_xyz.cast<double>();
        // Preserve finite long-range returns in the raw SLAM archive.  The
        // installed pipeline deskews every such ray (and therefore advances
        // RawImuTracker at its firing time) before the 60 m range stage drops
        // it from scan-matching geometry.  Filtering here changes the IMU
        // integration partition even though the endpoint never reaches ICP.
        // SurveyorSLAM converts the static sensor-frame transform and the
        // calibrated Pandar endpoint to float before constructing RangeData.
        // This is intentionally different from CloudBuilder's trajectory
        // path: retaining a double pose here changes most base_link endpoints
        // by a few float ULPs before IMU deskew.
        const Eigen::Vector3f point =
            rig_from_sensor.rotation.cast<float>() *
                scan.points[index].point.xyz +
            rig_from_sensor.translation.cast<float>();
        if (rejectedBySurveyorMinimumRangeFilter(scan.sensor, filter_point) ||
            rejectedBySurveyorSelfFilter(
                scan.sensor, filter_point,
                scan.points[index].point.raw_intensity)) {
            continue;
        }
        accepted.push_back(SlamScanPointNs{
            point.x(), point.y(), point.z(),
            // The ROS/PCL adapter subtracts the two Unix-epoch ``toSec()``
            // doubles and truncates that duration to integer nanoseconds.
            // This is observably different from subtracting the original
            // int64 stamps: on the frozen G11 stream 236/712 packet times
            // move by one nanosecond, exactly matching ImuTracker::Advance.
            collator_base_ns + static_cast<std::int64_t>(
                (scan.points[index].packet_timestamp - scan.header_timestamp) *
                1.0e9)});
    }
    if (accepted.size() > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("SLAM scan contains too many accepted points");
    }
    const std::uint32_t count = static_cast<std::uint32_t>(accepted.size());
    const std::uint32_t raw_ray_count =
        static_cast<std::uint32_t>(raw_ray_count_value);
    const std::uint32_t packet_count =
        static_cast<std::uint32_t>(packet_timestamps_ns.size());
    output.write(reinterpret_cast<const char*>(&scan.sensor), sizeof(scan.sensor));
    output.write(
        reinterpret_cast<const char*>(&scan.header_timestamp_ns),
        sizeof(scan.header_timestamp_ns));
    output.write(reinterpret_cast<const char*>(&raw_ray_count), sizeof(raw_ray_count));
    output.write(reinterpret_cast<const char*>(&packet_count), sizeof(packet_count));
    output.write(reinterpret_cast<const char*>(&count), sizeof(count));
    output.write(
        reinterpret_cast<const char*>(packet_timestamps_ns.data()),
        static_cast<std::streamsize>(packet_timestamps_ns.size() *
                                     sizeof(std::int64_t)));
    output.write(
        reinterpret_cast<const char*>(accepted.data()),
        static_cast<std::streamsize>(accepted.size() * sizeof(SlamScanPointNs)));
    if (!output) {
        throw std::runtime_error("failed while writing SLAM scan stream");
    }
    scan = BufferedPandarScan{};
    return input_count;
}

int runSlamScanExtraction(const Options& options, const SteadyClock::time_point total_started) {
    if (!options.slam_scans_output.parent_path().empty()) {
        fs::create_directories(options.slam_scans_output.parent_path());
    }
    std::ofstream output(options.slam_scans_output, std::ios::binary | std::ios::trunc);
    if (!output) {
        throw std::runtime_error(
            "cannot open SLAM scan output: " + options.slam_scans_output.string());
    }
    constexpr std::array<char, 8> magic{{'N', 'V', 'S', 'L', 'A', 'M', '6', '\0'}};
    output.write(magic.data(), static_cast<std::streamsize>(magic.size()));

    CloudBuilderTiming timing;
    BufferedPandarScan pending_scan;
    std::uint64_t packets = 0U;
    std::uint64_t scans = 0U;
    std::uint64_t returns = 0U;
    std::uint64_t fringe_rejected = 0U;
    const auto flush = [&]() {
        if (!pending_scan.active) {
            return;
        }
        returns += flushPandarSlamScan(
            pending_scan, options, output, fringe_rejected, timing);
        ++scans;
    };
    while (true) {
        std::uint8_t sensor = 0U;
        double timestamp = 0.0;
        std::int64_t timestamp_ns = 0;
        std::uint16_t size = 0U;
        if (!std::cin.read(reinterpret_cast<char*>(&sensor), sizeof(sensor))) {
            break;
        }
        if (options.frame_timestamps_ns) {
            if (!std::cin.read(
                    reinterpret_cast<char*>(&timestamp_ns), sizeof(timestamp_ns))) {
                throw std::runtime_error("truncated packet frame timestamp");
            }
            timestamp = rosSecondsFromNanoseconds(timestamp_ns);
        } else if (!std::cin.read(
                       reinterpret_cast<char*>(&timestamp), sizeof(timestamp))) {
            throw std::runtime_error("truncated packet frame timestamp");
        } else {
            timestamp_ns = static_cast<std::int64_t>(
                std::llround(timestamp * 1.0e9));
        }
        if (!std::cin.read(reinterpret_cast<char*>(&size), sizeof(size))) {
            throw std::runtime_error("truncated packet frame header");
        }
        if (size == 0U) {
            flush();
            pending_scan.sensor = sensor;
            pending_scan.header_timestamp = timestamp;
            pending_scan.header_timestamp_ns = timestamp_ns;
            pending_scan.active = true;
            continue;
        }
        std::vector<std::uint8_t> data(size);
        if (!std::cin.read(reinterpret_cast<char*>(data.data()), size)) {
            throw std::runtime_error("truncated packet payload");
        }
        if (size != kPacketBytes) {
            throw std::runtime_error(
                "SLAM scan extraction supports Pandar XTM packets only");
        }
        if (!pending_scan.active || pending_scan.sensor != sensor) {
            flush();
            pending_scan.sensor = sensor;
            pending_scan.header_timestamp = timestamp;
            pending_scan.header_timestamp_ns = timestamp_ns;
            pending_scan.active = true;
        }
        appendPandarPacket(
            pending_scan, std::move(data), timestamp, timestamp_ns);
        ++packets;
    }
    flush();
    output.close();
    std::cerr << "SLAM scan stream: packets=" << packets << "; scans=" << scans
              << "; returns before fringe=" << returns
              << "; fringe removed=" << fringe_rejected
              << "; decode=" << timing.decode << " s; order/normals="
              << timing.order_and_normals << " s; fringe=" << timing.fringe
              << " s; total=" << elapsedSeconds(total_started) << " s\n";
    return 0;
}

std::size_t flushPandarScan(
    BufferedPandarScan& scan, const Options& options,
    const std::vector<PrecisePose>& trajectory, StreamingVoxelCloud& output,
    UnvoxelizedCloudWriter& unvoxelized_output,
    std::array<std::optional<CloudBuilderPose>, 2>& previous_accepted_scan_pose,
    std::array<std::optional<CloudBuilderPose>, 2>& previous_observed_scan_pose,
    std::array<std::uint64_t, 2>& accepted_scan_count,
    std::uint64_t& motion_rejected_scans, std::uint64_t& fringe_rejected,
    std::uint64_t& foot_rejected,
    std::ofstream* scan_stats, CloudBuilderTiming& timing) {
    if (!scan.active) {
        return 0U;
    }
    decodePandarScan(scan, options, timing);
    const std::size_t input_count = scan.points.size();
    if (scan.points.empty()) {
        scan = BufferedPandarScan{};
        return input_count;
    }
    // cloud_builder enables --filter-no-motion by default. Each laser topic
    // maintains its own last accepted scan pose; rejected scans do not advance
    // that pose, so sub-threshold movement accumulates until either the 5 mm
    // translation or 0.05 degree rotation threshold is crossed.
    // The installed filter consumes the same uint64_t/Rigid3<float>
    // TrajectoryProvider as point unskewing. Running this decision on the
    // original double trajectory moves a small number of scans across the
    // 5 mm / 0.05 degree thresholds over a complete recording.
    // The motion filter is attached after the per-topic sensor transform.  Its
    // pose therefore follows the laser origin, not the rig origin.  The lever
    // arm matters at the 5 mm boundary whenever the rig rotates between two
    // otherwise almost stationary scans.
    const PrecisePose& rig_from_sensor =
        scan.sensor == 0U ? options.rig_from_horiz : options.rig_from_vert;
    const CloudBuilderPose scan_pose = composeCloudBuilderPose(
        poseAtCloudBuilder(trajectory, scan.header_timestamp_ns / 1000),
        rig_from_sensor);
    auto& previous_pose = previous_accepted_scan_pose[scan.sensor];
    auto& previous_observed_pose = previous_observed_scan_pose[scan.sensor];
    const auto relative_motion =
        [&scan_pose](const std::optional<CloudBuilderPose>& previous) {
        if (!previous) {
            return std::pair<double, double>{
                std::numeric_limits<double>::quiet_NaN(),
                std::numeric_limits<double>::quiet_NaN()};
        }
        const float translation =
            (scan_pose.translation - previous->translation).norm();
        const float angle_degrees = Eigen::AngleAxisf(
            previous->rotation.conjugate() * scan_pose.rotation).angle() *
            180.0F / kPi;
        return std::pair<double, double>{
            static_cast<double>(translation), static_cast<double>(angle_degrees)};
    };
    const auto [accepted_translation, accepted_angle] = relative_motion(previous_pose);
    const auto [observed_translation, observed_angle] = relative_motion(previous_observed_pose);
    previous_observed_pose = scan_pose;
    bool motion_rejected = false;
    if (options.no_motion_filter &&
        accepted_scan_count[scan.sensor] >= 2U && previous_pose) {
        if (accepted_translation < kNoMotionMaximumTranslation &&
            accepted_angle < kNoMotionMaximumAngleDegrees) {
            if (scan.insertion_enabled) {
                ++motion_rejected_scans;
            }
            motion_rejected = true;
        }
    }
    const auto write_scan_stats = [&](std::uint64_t scan_fringe_rejected,
                                      std::uint64_t scan_foot_rejected) {
        if (scan_stats == nullptr) {
            return;
        }
        *scan_stats << std::setprecision(17) << scan.header_timestamp << ','
                    << static_cast<unsigned int>(scan.sensor) << ','
                    << (scan.insertion_enabled ? 1 : 0) << ',' << input_count << ','
                    << (motion_rejected ? 0 : 1) << ','
                    << accepted_translation << ',' << accepted_angle << ','
                    << observed_translation << ',' << observed_angle << ','
                    << scan_pose.translation.x() << ',' << scan_pose.translation.y() << ','
                    << scan_pose.translation.z() << ',' << scan_pose.rotation.x() << ','
                    << scan_pose.rotation.y() << ',' << scan_pose.rotation.z() << ','
                    << scan_pose.rotation.w() << ',' << scan_fringe_rejected << ','
                    << scan_foot_rejected << '\n';
    };
    if (motion_rejected) {
        write_scan_stats(0U, 0U);
        scan = BufferedPandarScan{};
        return 0U;
    }
    previous_pose = scan_pose;
    ++accepted_scan_count[scan.sensor];
    if (!scan.insertion_enabled) {
        write_scan_stats(0U, 0U);
        scan = BufferedPandarScan{};
        return 0U;
    }
    if (options.scan_stats_only) {
        write_scan_stats(0U, 0U);
        scan = BufferedPandarScan{};
        return input_count;
    }
    const std::int64_t scan_timestamp_us = scan.header_timestamp_ns / 1000;
    const CloudBuilderPose world_from_sensor_reference = composeCloudBuilderPose(
        poseAtCloudBuilder(trajectory, scan_timestamp_us), rig_from_sensor);
    const CloudBuilderPose sensor_reference_pose_from_world =
        inverseCloudBuilderPose(world_from_sensor_reference);
    const Eigen::Matrix4f sensor_reference_from_world =
        cloudBuilderPoseMatrix(sensor_reference_pose_from_world);

    std::vector<Vec3f> motion_compensated_coordinates(scan.points.size());
    const bool needs_foot_region =
        options.vertical_foot_filter && scan.sensor == 1U;
    std::vector<Vec3f> foot_region_coordinates(
        needs_foot_region ? scan.points.size() : 0U);
    std::vector<Vec3f> world_coordinates(scan.points.size());
    std::vector<Vec3f> world_origins(scan.points.size());
    std::vector<Eigen::Matrix4f> world_from_sensor_matrices;
    std::vector<Vec3f> world_origin_translations;
    std::vector<std::size_t> point_transform_indices(scan.points.size());
    world_from_sensor_matrices.reserve(scan.packets.size());
    world_origin_translations.reserve(scan.packets.size());
    std::int64_t previous_query_us = std::numeric_limits<std::int64_t>::min();
    std::int64_t previous_origin_query_us =
        std::numeric_limits<std::int64_t>::min();
    for (std::size_t index = 0; index < scan.points.size(); ++index) {
        const BufferedPandarPoint& buffered = scan.points[index];
        // The reference mixes two ROS time representations here: the scan
        // base is integer microseconds, while the packet offset is float(
        // packet.stamp.toSec() - scan.header.stamp.toSec()). Preserve the two
        // epoch-double conversions; using the mathematically exact nanosecond
        // difference changes a few provider queries by one microsecond.
        const float relative_seconds = static_cast<float>(
            buffered.packet_timestamp - scan.header_timestamp);
        // cloud_builder converts the integer scan timestamp to double, adds
        // the float packet offset in microseconds, and only then truncates the
        // sum back to int64. At Unix-epoch magnitudes the double ULP is 0.25 us,
        // so truncating the relative term first changes a tiny number of
        // boundary points by one microsecond.
        const std::int64_t query_us = static_cast<std::int64_t>(
            static_cast<double>(scan_timestamp_us) +
            static_cast<double>(relative_seconds) * 1.0e6);
        // PointRayIntensity is assembled later than the motion-compensated
        // endpoint. That adapter performs its timestamp conversion in a
        // different order: cvttss2si(relative * 1e6F), then integer-adds the
        // scan base. Keep this second provider query; folding the two paths
        // together changes ray origins by up to one trajectory microsecond.
        const std::int64_t origin_query_us = scan_timestamp_us +
            static_cast<std::int64_t>(relative_seconds * 1.0e6F);
        if (query_us != previous_query_us ||
            origin_query_us != previous_origin_query_us) {
            world_from_sensor_matrices.push_back(cloudBuilderPoseMatrix(
                composeCloudBuilderPose(
                    poseAtCloudBuilder(trajectory, query_us), rig_from_sensor)));
            world_origin_translations.push_back(composeCloudBuilderPose(
                poseAtCloudBuilder(trajectory, origin_query_us),
                rig_from_sensor).translation);
            previous_query_us = query_us;
            previous_origin_query_us = origin_query_us;
        }
        point_transform_indices[index] = world_from_sensor_matrices.size() - 1U;
    }
    const auto transform_started = SteadyClock::now();
#pragma omp parallel for schedule(static)
    for (std::size_t index = 0; index < scan.points.size(); ++index) {
        const BufferedPandarPoint& buffered = scan.points[index];
        const Eigen::Matrix4f& world_from_sensor =
            world_from_sensor_matrices[point_transform_indices[index]];
        const Vec3f point_world =
            applyCloudBuilderMatrix(world_from_sensor, buffered.point.xyz);
        world_coordinates[index] = point_world;
        world_origins[index] =
            world_origin_translations[point_transform_indices[index]];
        motion_compensated_coordinates[index] =
            applyCloudBuilderMatrix(sensor_reference_from_world, point_world);
        // PlaneFilter's Point/Region adapter takes a separate path from the
        // ordered cloud used by the fringe filter: it applies the inverse
        // Rigid3f directly with Eigen's quaternion-vector kernel. Materializing
        // a Matrix4f changes cylinder-boundary coordinates by several ULPs.
        if (needs_foot_region) {
            foot_region_coordinates[index] =
                sensor_reference_pose_from_world.rotation * point_world +
                sensor_reference_pose_from_world.translation;
        }
    }
    timing.transform += elapsedSeconds(transform_started);
    const auto order_started = SteadyClock::now();
    OrderedPandarScan ordered = orderPandarScan(
        scan.points, std::move(motion_compensated_coordinates));
    timing.order_and_normals += elapsedSeconds(order_started);
    std::vector<std::uint8_t> rejected(scan.points.size(), 0U);
    std::uint64_t scan_fringe_rejected = 0U;
    const auto fringe_started = SteadyClock::now();
    if (options.multilayer_fringe_filter) {
#pragma omp parallel for schedule(static) reduction(+:scan_fringe_rejected)
        for (int ring = 0; ring < kRings; ++ring) {
            for (std::size_t column = 0; column < ordered.rows[ring].size(); ++column) {
                const std::size_t index = ordered.rows[ring][column];
                if (isMultilayerFringe(ordered, ring, column)) {
                    rejected[index] = 1U;
                    ++scan_fringe_rejected;
                }
            }
        }
        fringe_rejected += scan_fringe_rejected;
    }
    timing.fringe += elapsedSeconds(fringe_started);
    std::uint64_t scan_foot_rejected = 0U;
    const auto foot_started = SteadyClock::now();
    if (needs_foot_region) {
        std::vector<std::size_t> foot_candidates;
        foot_candidates.reserve(scan.points.size() / 6U);
        for (std::size_t index = 0; index < scan.points.size(); ++index) {
            if (rejected[index]) {
                continue;
            }
            // PlaneFilter asks its RegionFilter to evaluate the point in the
            // scan-reference frame produced by the Point/Region adapter above.
            // Its nested RegionBoolean is cylinder(r=0.5 m) AND z<=0.
            // Use the serialized float matrix from RegionTransformed. Rebuilding
            // it from the sensor-frame quaternion changes several coefficients
            // by a few ULPs and moves exact cylinder-boundary returns inside.
            const Vec3f& point = foot_region_coordinates[index];
            const Vec3f region_point{
                (-0.006795319262892008F * point.x() +
                    0.11973387002944946F * point.y()) +
                    0.9927828907966614F * point.z(),
                (-0.9999668598175049F * point.x() +
                    0.0036559293512254953F * point.y()) -
                    0.007285381201654673F * point.z(),
                (-0.004501895513385534F * point.x() -
                    0.9927994608879089F * point.y()) +
                    0.11970502138137817F * point.z()};
            // RegionCylindrical compares radiusSquared >= x*x+y*y (setae), so
            // the boundary is inclusive.
            if (region_point.x() * region_point.x() +
                    region_point.y() * region_point.y() <=
                    kFootRegionRadiusSquared &&
                region_point.z() <= 0.0F) {
                foot_candidates.push_back(index);
            }
        }
        if (const auto plane = fitFootPlaneRansac(
                world_coordinates, foot_candidates)) {
            for (const std::size_t index : foot_candidates) {
                const Vec3f& point = world_coordinates[index];
                const float distance = std::abs(
                    ((*plane).z() * point.z() + (*plane).y() * point.y()) +
                    (*plane).x() * point.x() + (*plane).w());
                if (distance > kFootMaximumPlaneDistance) {
                    rejected[index] = 1U;
                    ++scan_foot_rejected;
                    ++foot_rejected;
                }
            }
        }
    }
    timing.foot += elapsedSeconds(foot_started);
    write_scan_stats(scan_fringe_rejected, scan_foot_rejected);
    if (options.filter_stats_only) {
        scan = BufferedPandarScan{};
        return input_count;
    }
    const auto emit_started = SteadyClock::now();
    for (std::size_t index = 0; index < scan.points.size(); ++index) {
        if (rejected[index]) {
            continue;
        }
        const BufferedPandarPoint& buffered = scan.points[index];
        const Vec3f normal = ordered.normals[index].allFinite() ?
            ordered.normals[index] : Vec3f::Zero();
            emitTransformedPandarPoint(
                buffered.point, normal, scan.sensor, buffered.packet_timestamp,
                options, world_coordinates[index], world_origins[index],
                world_from_sensor_reference.rotation, output, unvoxelized_output);
    }
    timing.emit += elapsedSeconds(emit_started);
    scan = BufferedPandarScan{};
    return input_count;
}

PacketFormat processVelodynePacket(
    const std::vector<std::uint8_t>& data, std::uint8_t sensor, double timestamp,
    const Options& options, const std::vector<PrecisePose>& trajectory,
    StreamingVoxelCloud& output,
    UnvoxelizedCloudWriter& unvoxelized_output) {
    if (!validVelodynePacket(data)) {
        throw std::runtime_error("invalid 1206-byte Velodyne packet structure");
    }
    const auto decoded = decodeVelodynePacket(data, sensor, options);
    for (const auto& point : decoded) {
        emitPoint(point, Vec3f::Zero(), sensor, timestamp, options, trajectory, output,
                  unvoxelized_output);
    }
    return PacketFormat::VelodyneVlp16;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        const auto total_started = SteadyClock::now();
        const Options options = parseArguments(argc, argv);
        if (!options.slam_scans_output.empty()) {
            return runSlamScanExtraction(options, total_started);
        }
        const auto trajectory = readTrajectory(options.trajectory);
        CloudBuilderTiming timing;
        StreamingVoxelCloud cloud(options, timing);
        UnvoxelizedCloudWriter unvoxelized_cloud(options.unvoxelized_output);
        std::uint64_t packets = 0U;
        std::uint64_t pandar_packets = 0U;
        std::uint64_t velodyne_packets = 0U;
        std::uint64_t pandar_scans = 0U;
        std::uint64_t pandar_returns_before_fringe = 0U;
        std::uint64_t motion_rejected_scans = 0U;
        std::uint64_t fringe_rejected = 0U;
        std::uint64_t foot_rejected = 0U;
        std::array<std::optional<CloudBuilderPose>, 2> previous_accepted_scan_pose;
        std::array<std::optional<CloudBuilderPose>, 2> previous_observed_scan_pose;
        std::array<std::uint64_t, 2> accepted_scan_count{{0U, 0U}};
        std::ofstream scan_stats;
        if (!options.scan_stats.empty()) {
            scan_stats.open(options.scan_stats);
            if (!scan_stats) {
                throw std::runtime_error(
                    "cannot open scan statistics: " + options.scan_stats.string());
            }
            scan_stats << "timestamp,sensor,insertion_enabled,input_count,accepted,"
                          "translation_from_accepted,angle_from_accepted_deg,"
                          "translation_from_observed,angle_from_observed_deg,"
                          "tx,ty,tz,qx,qy,qz,qw,fringe_rejected,foot_rejected\n";
        }
        std::ofstream* scan_stats_output = scan_stats.is_open() ? &scan_stats : nullptr;
        BufferedPandarScan pending_scan;
        while (true) {
            std::uint8_t sensor = 0U;
            double timestamp = 0.0;
            std::int64_t timestamp_ns = 0;
            std::uint16_t size = 0U;
            if (!std::cin.read(reinterpret_cast<char*>(&sensor), sizeof(sensor))) {
                break;
            }
            if (options.frame_timestamps_ns) {
                if (!std::cin.read(
                        reinterpret_cast<char*>(&timestamp_ns), sizeof(timestamp_ns))) {
                    throw std::runtime_error("truncated packet frame timestamp");
                }
                timestamp = rosSecondsFromNanoseconds(timestamp_ns);
            } else if (!std::cin.read(
                           reinterpret_cast<char*>(&timestamp), sizeof(timestamp))) {
                throw std::runtime_error("truncated packet frame timestamp");
            } else {
                timestamp_ns = static_cast<std::int64_t>(
                    std::llround(timestamp * 1.0e9));
            }
            if (!std::cin.read(reinterpret_cast<char*>(&size), sizeof(size))) {
                throw std::runtime_error("truncated packet frame header");
            }
            if (size == 0U) {
                if (pending_scan.active) {
                    const bool completed_scan_was_enabled =
                        pending_scan.insertion_enabled;
                    pandar_returns_before_fringe += flushPandarScan(
                        pending_scan, options, trajectory, cloud,
                        unvoxelized_cloud, previous_accepted_scan_pose,
                        previous_observed_scan_pose, accepted_scan_count,
                        motion_rejected_scans, fringe_rejected, foot_rejected,
                        scan_stats_output, timing);
                    pandar_scans += completed_scan_was_enabled ? 1U : 0U;
                }
                const bool insertion_enabled = (sensor & 0x80U) == 0U;
                sensor &= 0x7fU;
                if (sensor > 1U) {
                    throw std::runtime_error("invalid sensor id in scan marker");
                }
                pending_scan.sensor = sensor;
                pending_scan.header_timestamp = timestamp;
                pending_scan.header_timestamp_ns = timestamp_ns;
                pending_scan.insertion_enabled = insertion_enabled;
                pending_scan.active = true;
                continue;
            }
            std::vector<std::uint8_t> data(size);
            if (!std::cin.read(reinterpret_cast<char*>(data.data()), size)) {
                throw std::runtime_error("truncated packet payload");
            }
            if (size == kPacketBytes) {
                if (!pending_scan.active || pending_scan.sensor != sensor) {
                    if (pending_scan.active) {
                        const bool completed_scan_was_enabled =
                            pending_scan.insertion_enabled;
                        pandar_returns_before_fringe += flushPandarScan(
                            pending_scan, options, trajectory, cloud,
                            unvoxelized_cloud, previous_accepted_scan_pose,
                            previous_observed_scan_pose, accepted_scan_count,
                            motion_rejected_scans, fringe_rejected, foot_rejected,
                            scan_stats_output, timing);
                        pandar_scans += completed_scan_was_enabled ? 1U : 0U;
                    }
                    pending_scan.sensor = sensor;
                    pending_scan.header_timestamp = timestamp;
                    pending_scan.header_timestamp_ns = timestamp_ns;
                    pending_scan.active = true;
                }
                appendPandarPacket(
                    pending_scan, std::move(data), timestamp, timestamp_ns);
                ++pandar_packets;
            } else if (size == kVelodynePacketBytes) {
                if (pending_scan.active) {
                    const bool completed_scan_was_enabled =
                        pending_scan.insertion_enabled;
                    pandar_returns_before_fringe += flushPandarScan(
                        pending_scan, options, trajectory, cloud,
                        unvoxelized_cloud, previous_accepted_scan_pose,
                        previous_observed_scan_pose, accepted_scan_count,
                        motion_rejected_scans, fringe_rejected, foot_rejected,
                        scan_stats_output, timing);
                    pandar_scans += completed_scan_was_enabled ? 1U : 0U;
                }
                processVelodynePacket(data, sensor, timestamp, options, trajectory,
                                      cloud, unvoxelized_cloud);
                ++velodyne_packets;
            } else {
                throw std::runtime_error(
                    "unsupported laser packet size: " + std::to_string(data.size()));
            }
            if (++packets % 10000U == 0U) {
                std::cerr << "Processed " << packets << " laser packets\n";
            }
        }
        if (pending_scan.active) {
            const bool completed_scan_was_enabled =
                pending_scan.insertion_enabled;
            pandar_returns_before_fringe += flushPandarScan(
                pending_scan, options, trajectory, cloud,
                unvoxelized_cloud, previous_accepted_scan_pose,
                previous_observed_scan_pose, accepted_scan_count,
                motion_rejected_scans, fringe_rejected, foot_rejected,
                scan_stats_output, timing);
            pandar_scans += completed_scan_was_enabled ? 1U : 0U;
        }
        cloud.finish();
        unvoxelized_cloud.finish();
        std::cerr << "Decoded packet formats: Pandar XTM=" << pandar_packets
                  << "; Velodyne VLP16=" << velodyne_packets << '\n';
        std::cerr << "Complete Pandar scans=" << pandar_scans
                  << "; no-motion scans removed=" << motion_rejected_scans
                  << "; returns before multilayer fringe=" << pandar_returns_before_fringe
                  << "; multilayer fringe returns removed=" << fringe_rejected
                  << "; vertical foot returns removed=" << foot_rejected << '\n';
        std::cerr << "CloudBuilder phase timing: decode " << timing.decode
                  << " s; transform " << timing.transform
                  << " s; order/normals " << timing.order_and_normals
                  << " s; fringe " << timing.fringe
                  << " s; foot " << timing.foot
                  << " s; emit/pack including in-loop shard flush " << timing.emit
                  << " s; shard partition " << timing.shard_partition
                  << " s; shard write " << timing.shard_write
                  << " s; retain " << timing.retain
                  << " s; total " << elapsedSeconds(total_started) << " s\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "navvis_recon_pandar: " << error.what() << '\n';
        return 1;
    }
}
