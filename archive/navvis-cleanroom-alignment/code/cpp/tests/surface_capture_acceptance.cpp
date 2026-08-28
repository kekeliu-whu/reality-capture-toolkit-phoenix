#include "navvis_recon/binary_surface_pipeline.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

#pragma pack(push, 1)
struct CapturedPoint48 {
    float field[12];
};

struct CapturedRawStatus52 {
    float field[12];
    std::uint32_t status;
};

struct CapturedVoxelPair24 {
    std::uint64_t first;
    std::uint64_t second;
    std::uint64_t selected;
};
#pragma pack(pop)
static_assert(sizeof(CapturedPoint48) == 48U);
static_assert(sizeof(CapturedRawStatus52) == 52U);
static_assert(sizeof(CapturedVoxelPair24) == 24U);

std::vector<navvis_recon::BinarySurfacePoint> readSurfacePoints(
    const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture " + path);
    std::vector<navvis_recon::BinarySurfacePoint> points;
    CapturedPoint48 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        points.push_back({
            navvis_recon::Vec3f(record.field[4], record.field[5], record.field[6]),
            navvis_recon::Vec3f(record.field[0], record.field[1], record.field[2]),
            record.field[8], record.field[9], record.field[10]});
    }
    return points;
}

std::vector<navvis_recon::BinarySurfacePoint> readVoxelAccumulators(
    const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open accumulator capture " + path);
    std::vector<navvis_recon::BinarySurfacePoint> points;
    float field[9]{};
    while (input.read(reinterpret_cast<char*>(field), sizeof(field))) {
        points.push_back({
            navvis_recon::Vec3f(field[0], field[1], field[2]),
            navvis_recon::Vec3f(field[3], field[4], field[5]),
            field[7], field[8], field[6]});
    }
    return points;
}

std::vector<navvis_recon::BinarySurfaceInput> readSurfaceInputs(
    const std::string& path) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture " + path);
    std::vector<navvis_recon::BinarySurfaceInput> points;
    CapturedPoint48 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        points.push_back({
            navvis_recon::Vec3f(record.field[4], record.field[5], record.field[6]),
            navvis_recon::Vec3f(record.field[0], record.field[1], record.field[2]),
            record.field[8], record.field[9]});
    }
    return points;
}

std::vector<navvis_recon::BinarySurfaceInput> readRawStatusInputs(
    const std::string& path, std::vector<std::uint8_t>* statuses = nullptr) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("cannot open capture " + path);
    std::vector<navvis_recon::BinarySurfaceInput> rays;
    CapturedRawStatus52 record{};
    while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
        rays.push_back({
            navvis_recon::Vec3f(record.field[4], record.field[5], record.field[6]),
            navvis_recon::Vec3f(record.field[0], record.field[1], record.field[2]),
            record.field[8], 1.0F});
        if (statuses != nullptr) {
            statuses->push_back(static_cast<std::uint8_t>(record.status));
        }
    }
    return rays;
}

}  // namespace

int main(int argc, char** argv) {
    try {
        if (argc < 2 || argc > 5) {
            throw std::invalid_argument(
                "usage: navvis_recon_surface_capture_acceptance "
                "[--normal INPUT.bin EXPECTED.bin|--selection INPUT.bin EXPECTED.bin|"
                "--smoothing SUPPORT.bin TARGET.bin EXPECTED_SUPPORT.bin|"
                "--post TARGET.bin SUPPORT.bin EXPECTED_TARGET.bin|"
                "--density-compare INPUT.bin EXPECTED.bin|"
                "--sor-compare INPUT.bin EXPECTED.bin|"
                "--voxel-primary INPUT.bin EXPECTED_ACCUMULATORS36.bin|"
                "--voxel-pairs INPUT.bin EXPECTED_PAIRS24.bin|"
                "--full-compare INPUT.bin EXPECTED.bin|"
                "--helper-input RAW_STATUS52.bin EXPECTED_INPUT48.bin|"
                "--helper-selection RAW_STATUS52.bin EXPECTED_SELECTED48.bin|"
                "--helper-selection-aggregated INPUT48.bin EXPECTED_SELECTED48.bin|"
                "--helper RAW_STATUS52.bin EXPECTED_HELPER48.bin|"
                "--helper-aggregated INPUT48.bin EXPECTED_HELPER48.bin|"
                "--occlusion-helper RAW_STATUS52.bin HELPER48.bin|"
                "--occlusion-indices RAW48.bin HELPER48.bin INDEX[,INDEX...]|"
                "--occlusion-input-indices RAW48.bin HELPER_INPUT48.bin INDEX[,INDEX...]|"
                "--occlusion-main-input RAW_STATUS52.bin EXPECTED_INPUT48.bin|"
                "--occlusion RAW_STATUS52.bin|"
                "--voxel|--density|--sor] CAPTURE.bin");
        }
        const std::string mode = argc >= 3 ? argv[1] : "--full";
        if (mode == "--full-compare") {
            if (argc != 4) {
                throw std::invalid_argument("--full-compare requires input and expected captures");
            }
            const auto input = readSurfaceInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            navvis_recon::BinarySurfaceStageCounts counts;
            const auto output = navvis_recon::runBinarySurfacePipeline(
                input, navvis_recon::BinarySurfaceOptions{}, &counts);
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t record_bit_exact = 0U;
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t curvature_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz = std::memcmp(
                    output[index].xyz.data(), expected[index].xyz.data(), 12U) == 0;
                const bool normal = std::memcmp(
                    output[index].normal.data(), expected[index].normal.data(), 12U) == 0;
                const bool intensity = std::memcmp(
                    &output[index].intensity, &expected[index].intensity, 4U) == 0;
                const bool curvature = std::memcmp(
                    &output[index].curvature, &expected[index].curvature, 4U) == 0;
                const bool weight = std::memcmp(
                    &output[index].weight, &expected[index].weight, 4U) == 0;
                xyz_bit_exact += xyz;
                normal_bit_exact += normal;
                intensity_bit_exact += intensity;
                curvature_bit_exact += curvature;
                weight_bit_exact += weight;
                record_bit_exact += xyz && normal && intensity && curvature && weight;
            }
            std::cout << "{\"stage\":\"full-compare\",\"input\":" << input.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"record_bit_exact\":" << record_bit_exact
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"curvature_bit_exact\":" << curvature_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"counts\":{\"valid\":" << counts.valid
                      << ",\"voxel\":" << counts.output_voxels
                      << ",\"density\":" << counts.density
                      << ",\"sor\":" << counts.adaptive_sor
                      << "}}\n";
            return 0;
        }
        if (mode == "--voxel-pairs") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--voxel-pairs requires input and expected pair captures");
            }
            const auto input = readSurfacePoints(argv[2]);
            std::ifstream capture(argv[3], std::ios::binary);
            if (!capture) throw std::runtime_error("cannot open pair capture");
            std::vector<CapturedVoxelPair24> expected;
            CapturedVoxelPair24 pair{};
            while (capture.read(reinterpret_cast<char*>(&pair), sizeof(pair))) {
                expected.push_back(pair);
            }
            const auto output = navvis_recon::inspectBinaryOutputVoxelMergeChoices(input);
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t exact = 0U;
            std::size_t selected = 0U;
            std::size_t expected_selected = 0U;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool output_selected =
                    output[index] != std::numeric_limits<std::uint32_t>::max();
                const bool capture_selected = expected[index].selected != 0U;
                selected += output_selected;
                expected_selected += capture_selected;
                exact += expected[index].first == index &&
                         output_selected == capture_selected &&
                         (!output_selected || output[index] == expected[index].second);
            }
            std::cout << "{\"stage\":\"voxel-pairs\",\"input\":" << input.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"exact\":" << exact
                      << ",\"expected_selected\":" << expected_selected
                      << ",\"selected\":" << selected << "}\n";
            return 0;
        }
        if (mode == "--voxel-primary") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--voxel-primary requires input and expected accumulator captures");
            }
            const auto input = readSurfacePoints(argv[2]);
            const auto expected = readVoxelAccumulators(argv[3]);
            const auto output =
                navvis_recon::applyBinaryOutputVoxelPrimaryAggregation(input);
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t record_bit_exact = 0U;
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t curvature_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz = std::memcmp(
                    output[index].xyz.data(), expected[index].xyz.data(), 12U) == 0;
                const bool normal = std::memcmp(
                    output[index].normal.data(), expected[index].normal.data(), 12U) == 0;
                const bool intensity = std::memcmp(
                    &output[index].intensity, &expected[index].intensity, 4U) == 0;
                const bool curvature = std::memcmp(
                    &output[index].curvature, &expected[index].curvature, 4U) == 0;
                const bool weight = std::memcmp(
                    &output[index].weight, &expected[index].weight, 4U) == 0;
                xyz_bit_exact += xyz;
                normal_bit_exact += normal;
                intensity_bit_exact += intensity;
                curvature_bit_exact += curvature;
                weight_bit_exact += weight;
                record_bit_exact += xyz && normal && intensity && curvature && weight;
            }
            std::cout << "{\"stage\":\"voxel-primary\",\"input\":" << input.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"record_bit_exact\":" << record_bit_exact
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"curvature_bit_exact\":" << curvature_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact << "}\n";
            return 0;
        }
        if (mode == "--smoothing" || mode == "--post" ||
            mode == "--density-compare" || mode == "--sor-compare") {
            const bool has_secondary = mode == "--smoothing" || mode == "--post";
            if (argc != (has_secondary ? 5 : 4)) {
                throw std::invalid_argument(mode +
                    (has_secondary
                         ? " requires input, support/target, and expected captures"
                         : " requires input and expected captures"));
            }
            const auto first = readSurfacePoints(argv[2]);
            const auto second = readSurfacePoints(argv[3]);
            const auto expected = has_secondary ? readSurfacePoints(argv[4]) : second;
            std::vector<navvis_recon::BinarySurfacePoint> output;
            if (mode == "--smoothing") {
                output = navvis_recon::applyBinarySurfaceSupportPruning(first, second);
            } else if (mode == "--post") {
                output = navvis_recon::applyBinaryPostSmoothingFilter(first);
            } else if (mode == "--density-compare") {
                output = navvis_recon::applyBinaryDensityFilter(first);
            } else {
                output = navvis_recon::applyBinaryAdaptiveSor(first);
            }
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t record_bit_exact = 0U;
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t curvature_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            float xyz_maximum = 0.0F;
            float normal_maximum = 0.0F;
            float curvature_maximum = 0.0F;
            std::size_t mismatch_samples = 0U;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz_exact = std::memcmp(
                    output[index].xyz.data(), expected[index].xyz.data(), 12U) == 0;
                const bool normal_exact = std::memcmp(
                    output[index].normal.data(), expected[index].normal.data(), 12U) == 0;
                const bool intensity_exact = std::memcmp(
                    &output[index].intensity, &expected[index].intensity, 4U) == 0;
                const bool curvature_exact = std::memcmp(
                    &output[index].curvature, &expected[index].curvature, 4U) == 0;
                const bool weight_exact = std::memcmp(
                    &output[index].weight, &expected[index].weight, 4U) == 0;
                xyz_bit_exact += xyz_exact;
                normal_bit_exact += normal_exact;
                intensity_bit_exact += intensity_exact;
                curvature_bit_exact += curvature_exact;
                weight_bit_exact += weight_exact;
                record_bit_exact += xyz_exact && normal_exact && intensity_exact &&
                                    curvature_exact && weight_exact;
                xyz_maximum = std::max(xyz_maximum,
                    (output[index].xyz - expected[index].xyz).cwiseAbs().maxCoeff());
                normal_maximum = std::max(normal_maximum,
                    (output[index].normal - expected[index].normal).cwiseAbs().maxCoeff());
                curvature_maximum = std::max(curvature_maximum,
                    std::abs(output[index].curvature - expected[index].curvature));
                if (mode == "--post" && !curvature_exact && mismatch_samples < 16U) {
                    std::cerr << "post curvature mismatch index=" << index
                              << " output=" << output[index].curvature
                              << " expected=" << expected[index].curvature
                              << " xyz=" << output[index].xyz.transpose() << '\n';
                    ++mismatch_samples;
                }
            }
            std::cout << "{\"stage\":\"" << mode.substr(2)
                      << "\",\"input\":" << first.size()
                      << ",\"secondary\":" << second.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"record_bit_exact\":" << record_bit_exact
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"curvature_bit_exact\":" << curvature_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"xyz_maximum\":" << xyz_maximum
                      << ",\"normal_maximum\":" << normal_maximum
                      << ",\"curvature_maximum\":" << curvature_maximum << "}\n";
            return 0;
        }
        if (mode == "--helper-selection") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--helper-selection requires raw/status and expected captures");
            }
            const auto rays = readRawStatusInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            const auto helper_input =
                navvis_recon::applyBinaryOcclusionHelperInputAggregation(rays);
            navvis_recon::BinarySurfaceOptions options;
            options.minimum_cylinder_radius = 0.0141421352F;
            const auto normals =
                navvis_recon::applyBinaryMultiScaleNormalEstimation(helper_input, options);
            const auto output =
                navvis_recon::applyBinarySurfacePointSelection(normals, options);
            const std::size_t common = std::min(expected.size(), output.size());
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            std::size_t zero_mask_match = 0U;
            double xyz_sum = 0.0;
            double angle_sum = 0.0;
            float xyz_maximum = 0.0F;
            double angle_maximum = 0.0;
            for (std::size_t index = 0U; index < common; ++index) {
                xyz_bit_exact += std::memcmp(
                    output[index].xyz.data(), expected[index].xyz.data(),
                    3U * sizeof(float)) == 0 ? 1U : 0U;
                normal_bit_exact += std::memcmp(
                    output[index].normal.data(), expected[index].normal.data(),
                    3U * sizeof(float)) == 0 ? 1U : 0U;
                weight_bit_exact += std::memcmp(
                    &output[index].weight, &expected[index].weight,
                    sizeof(float)) == 0 ? 1U : 0U;
                const bool output_zero = output[index].normal.squaredNorm() == 0.0F;
                const bool expected_zero = expected[index].normal.squaredNorm() == 0.0F;
                zero_mask_match += output_zero == expected_zero ? 1U : 0U;
                const float xyz_error = (output[index].xyz - expected[index].xyz).norm();
                xyz_sum += xyz_error;
                xyz_maximum = std::max(xyz_maximum, xyz_error);
                if (!output_zero && !expected_zero) {
                    const double denominator = output[index].normal.cast<double>().norm() *
                        expected[index].normal.cast<double>().norm();
                    const double cosine = std::clamp(
                        output[index].normal.cast<double>().dot(
                            expected[index].normal.cast<double>()) / denominator,
                        -1.0, 1.0);
                    const double angle = 180.0 / 3.14159265358979323846 *
                        std::acos(cosine);
                    angle_sum += angle;
                    angle_maximum = std::max(angle_maximum, angle);
                }
            }
            std::cout << "{\"stage\":\"helper-selection\",\"raw_input\":"
                      << rays.size() << ",\"helper_input\":" << helper_input.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"zero_mask_match\":" << zero_mask_match
                      << ",\"xyz_mean\":" << (common ? xyz_sum / common : 0.0)
                      << ",\"xyz_maximum\":" << xyz_maximum
                      << ",\"angle_mean_degrees\":"
                      << (common ? angle_sum / common : 0.0)
                      << ",\"angle_maximum_degrees\":" << angle_maximum;
            if (!expected.empty() && !output.empty()) {
                std::uint32_t expected_normal_bits[3]{};
                std::uint32_t output_normal_bits[3]{};
                std::uint32_t expected_xyz_bits[3]{};
                std::uint32_t output_xyz_bits[3]{};
                std::memcpy(expected_normal_bits, expected.front().normal.data(), 12U);
                std::memcpy(output_normal_bits, output.front().normal.data(), 12U);
                std::memcpy(expected_xyz_bits, expected.front().xyz.data(), 12U);
                std::memcpy(output_xyz_bits, output.front().xyz.data(), 12U);
                std::cout << ",\"first_normal_bits\":[["
                          << expected_normal_bits[0] << ',' << expected_normal_bits[1]
                          << ',' << expected_normal_bits[2] << "],["
                          << output_normal_bits[0] << ',' << output_normal_bits[1]
                          << ',' << output_normal_bits[2] << "]]"
                          << ",\"first_xyz_bits\":[["
                          << expected_xyz_bits[0] << ',' << expected_xyz_bits[1]
                          << ',' << expected_xyz_bits[2] << "],["
                          << output_xyz_bits[0] << ',' << output_xyz_bits[1]
                          << ',' << output_xyz_bits[2] << "]]";
            }
            std::cout << "}\n";
            return 0;
        }
        if (mode == "--helper-selection-aggregated") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--helper-selection-aggregated requires helper input and expected captures");
            }
            const auto input = readSurfaceInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            navvis_recon::BinarySurfaceOptions options;
            options.minimum_cylinder_radius = 0.0141421352F;
            const auto normals =
                navvis_recon::applyBinaryMultiScaleNormalEstimation(input, options);
            const auto output =
                navvis_recon::applyBinarySurfacePointSelection(normals, options);
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t xyz_exact = 0U;
            std::size_t normal_exact = 0U;
            std::size_t intensity_exact = 0U;
            std::size_t curvature_exact = 0U;
            std::size_t weight_exact = 0U;
            std::vector<std::size_t> xyz_mismatches;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz_matches =
                    std::memcmp(output[index].xyz.data(), expected[index].xyz.data(),
                                3U * sizeof(float)) == 0;
                xyz_exact += xyz_matches;
                if (!xyz_matches) {
                    xyz_mismatches.push_back(index);
                }
                normal_exact +=
                    std::memcmp(output[index].normal.data(), expected[index].normal.data(),
                                3U * sizeof(float)) == 0;
                intensity_exact += std::memcmp(&output[index].intensity,
                                               &expected[index].intensity, sizeof(float)) == 0;
                curvature_exact += std::memcmp(&output[index].curvature,
                                               &expected[index].curvature, sizeof(float)) == 0;
                weight_exact += std::memcmp(&output[index].weight, &expected[index].weight,
                                            sizeof(float)) == 0;
            }
            std::cout << "{\"stage\":\"helper-selection-aggregated\",\"input\":"
                      << input.size() << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size() << ",\"xyz_bit_exact\":"
                      << xyz_exact << ",\"normal_bit_exact\":" << normal_exact
                      << ",\"intensity_bit_exact\":" << intensity_exact
                      << ",\"curvature_bit_exact\":" << curvature_exact
                      << ",\"weight_bit_exact\":" << weight_exact
                      << ",\"xyz_mismatches\":[";
            for (std::size_t sample = 0U; sample < xyz_mismatches.size(); ++sample) {
                if (sample != 0U) {
                    std::cout << ',';
                }
                const std::size_t index = xyz_mismatches[sample];
                std::uint32_t expected_bits[3]{};
                std::uint32_t output_bits[3]{};
                std::memcpy(expected_bits, expected[index].xyz.data(), sizeof(expected_bits));
                std::memcpy(output_bits, output[index].xyz.data(), sizeof(output_bits));
                std::cout << '[' << index << ",[" << expected_bits[0] << ','
                          << expected_bits[1] << ',' << expected_bits[2] << "],["
                          << output_bits[0] << ',' << output_bits[1] << ',' << output_bits[2]
                          << "]]";
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--helper-input") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--helper-input requires raw/status and expected captures");
            }
            const auto rays = readRawStatusInputs(argv[2]);
            const auto expected = readSurfaceInputs(argv[3]);
            const auto output =
                navvis_recon::applyBinaryOcclusionHelperInputAggregation(rays);
            const std::size_t common = std::min(expected.size(), output.size());
            std::size_t xyz_bit_exact = 0U;
            std::size_t origin_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            double xyz_sum = 0.0;
            double origin_sum = 0.0;
            for (std::size_t index = 0U; index < common; ++index) {
                xyz_bit_exact += std::memcmp(
                    expected[index].xyz.data(), output[index].xyz.data(),
                    3U * sizeof(float)) == 0 ? 1U : 0U;
                origin_bit_exact += std::memcmp(
                    expected[index].origin.data(), output[index].origin.data(),
                    3U * sizeof(float)) == 0 ? 1U : 0U;
                intensity_bit_exact += std::memcmp(
                    &expected[index].intensity, &output[index].intensity,
                    sizeof(float)) == 0 ? 1U : 0U;
                weight_bit_exact += std::memcmp(
                    &expected[index].weight, &output[index].weight,
                    sizeof(float)) == 0 ? 1U : 0U;
                xyz_sum += (expected[index].xyz - output[index].xyz).norm();
                origin_sum += (expected[index].origin - output[index].origin).norm();
            }
            std::cout << "{\"stage\":\"helper-input\",\"raw_input\":" << rays.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"origin_bit_exact\":" << origin_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"xyz_mean\":" << (common ? xyz_sum / common : 0.0)
                      << ",\"origin_mean\":" << (common ? origin_sum / common : 0.0)
                      << "}\n";
            return 0;
        }
        if (mode == "--helper") {
            if (argc != 4) {
                throw std::invalid_argument("--helper requires raw/status and expected captures");
            }
            const auto rays = readRawStatusInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            navvis_recon::BinaryOcclusionStageCounts counts;
            const auto output = navvis_recon::applyBinaryOcclusionHelperSurface(
                rays, {}, {}, &counts);
            using Key = std::array<int, 3>;
            std::map<Key, const navvis_recon::BinarySurfacePoint*> expected_by_key;
            std::map<Key, const navvis_recon::BinarySurfacePoint*> output_by_key;
            const auto key = [](const auto& point) {
                return Key{
                    static_cast<int>(std::floor(point.xyz.x() / 0.02F)),
                    static_cast<int>(std::floor(point.xyz.y() / 0.02F)),
                    static_cast<int>(std::floor(point.xyz.z() / 0.02F))};
            };
            for (const auto& point : expected) expected_by_key[key(point)] = &point;
            for (const auto& point : output) output_by_key[key(point)] = &point;
            std::size_t common_keys = 0U;
            std::size_t normal_direction_match = 0U;
            double xyz_sum = 0.0;
            double angle_sum = 0.0;
            float xyz_maximum = 0.0F;
            double angle_maximum = 0.0;
            for (const auto& [voxel, expected_point] : expected_by_key) {
                const auto found = output_by_key.find(voxel);
                if (found == output_by_key.end()) continue;
                ++common_keys;
                const auto* output_point = found->second;
                const float xyz_error = (output_point->xyz - expected_point->xyz).norm();
                xyz_sum += xyz_error;
                xyz_maximum = std::max(xyz_maximum, xyz_error);
                const double denominator = output_point->normal.cast<double>().norm() *
                    expected_point->normal.cast<double>().norm();
                if (denominator > 0.0) {
                    const double cosine = std::clamp(
                        output_point->normal.cast<double>().dot(
                            expected_point->normal.cast<double>()) / denominator,
                        -1.0, 1.0);
                    normal_direction_match += cosine > 0.0 ? 1U : 0U;
                    const double angle = 180.0 / 3.14159265358979323846 *
                        std::acos(cosine);
                    angle_sum += angle;
                    angle_maximum = std::max(angle_maximum, angle);
                }
            }
            std::cout << "{\"stage\":\"helper\",\"raw_input\":" << rays.size()
                      << ",\"helper_input\":" << counts.helper_input
                      << ",\"helper_valid\":" << counts.helper_valid
                      << ",\"helper_initial_voxels\":"
                      << counts.helper_initial_voxels
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"expected_unique_keys\":" << expected_by_key.size()
                      << ",\"output_unique_keys\":" << output_by_key.size()
                      << ",\"common_keys\":" << common_keys
                      << ",\"normal_direction_match\":" << normal_direction_match
                      << ",\"xyz_mean\":" << (common_keys ? xyz_sum / common_keys : 0.0)
                      << ",\"xyz_maximum\":" << xyz_maximum
                      << ",\"angle_mean_degrees\":"
                      << (common_keys ? angle_sum / common_keys : 0.0)
                      << ",\"angle_maximum_degrees\":" << angle_maximum;
            if (!expected.empty() && !output.empty()) {
                std::uint32_t expected_first_bits[3]{};
                std::uint32_t output_first_bits[3]{};
                std::memcpy(expected_first_bits, expected.front().xyz.data(), 12U);
                std::memcpy(output_first_bits, output.front().xyz.data(), 12U);
                std::cout << ",\"expected_first_xyz\":["
                          << expected.front().xyz.x() << ','
                          << expected.front().xyz.y() << ','
                          << expected.front().xyz.z() << ']'
                          << ",\"output_first_xyz\":["
                          << output.front().xyz.x() << ','
                          << output.front().xyz.y() << ','
                          << output.front().xyz.z() << ']'
                          << ",\"expected_first_bits\":["
                          << expected_first_bits[0] << ',' << expected_first_bits[1]
                          << ',' << expected_first_bits[2] << ']'
                          << ",\"output_first_bits\":["
                          << output_first_bits[0] << ',' << output_first_bits[1]
                          << ',' << output_first_bits[2] << ']';
            }
            std::cout << "}\n";
            return 0;
        }
        if (mode == "--helper-aggregated") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--helper-aggregated requires helper input and expected captures");
            }
            const auto input = readSurfaceInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            navvis_recon::BinaryOcclusionStageCounts counts;
            const auto output = navvis_recon::applyBinaryOcclusionHelperSurfaceFromInput(
                input, {}, {}, &counts);
            const std::size_t common = std::min(output.size(), expected.size());
            std::size_t xyz_exact = 0U;
            std::size_t normal_exact = 0U;
            std::size_t intensity_exact = 0U;
            std::size_t curvature_exact = 0U;
            std::size_t weight_exact = 0U;
            std::vector<std::size_t> xyz_mismatches;
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz_matches =
                    std::memcmp(output[index].xyz.data(), expected[index].xyz.data(),
                                3U * sizeof(float)) == 0;
                xyz_exact += xyz_matches;
                if (!xyz_matches) {
                    xyz_mismatches.push_back(index);
                }
                normal_exact +=
                    std::memcmp(output[index].normal.data(), expected[index].normal.data(),
                                3U * sizeof(float)) == 0;
                intensity_exact += std::memcmp(&output[index].intensity,
                                               &expected[index].intensity, sizeof(float)) == 0;
                curvature_exact += std::memcmp(&output[index].curvature,
                                               &expected[index].curvature, sizeof(float)) == 0;
                weight_exact += std::memcmp(&output[index].weight, &expected[index].weight,
                                            sizeof(float)) == 0;
            }
            std::cout << "{\"stage\":\"helper-aggregated\",\"input\":" << input.size()
                      << ",\"expected\":" << expected.size() << ",\"output\":"
                      << output.size() << ",\"xyz_bit_exact\":" << xyz_exact
                      << ",\"normal_bit_exact\":" << normal_exact
                      << ",\"intensity_bit_exact\":" << intensity_exact
                      << ",\"curvature_bit_exact\":" << curvature_exact
                      << ",\"weight_bit_exact\":" << weight_exact
                      << ",\"xyz_mismatches\":[";
            for (std::size_t sample = 0U; sample < xyz_mismatches.size(); ++sample) {
                if (sample != 0U) {
                    std::cout << ',';
                }
                const std::size_t index = xyz_mismatches[sample];
                std::cout << "[" << index << ",[" << expected[index].xyz.x() << ','
                          << expected[index].xyz.y() << ',' << expected[index].xyz.z()
                          << "],[" << output[index].xyz.x() << ',' << output[index].xyz.y()
                          << ',' << output[index].xyz.z() << "]]";
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--occlusion-helper") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--occlusion-helper requires raw/status and helper captures");
            }
            std::vector<std::uint8_t> expected;
            const auto rays = readRawStatusInputs(argv[2], &expected);
            const auto helper = readSurfacePoints(argv[3]);
            std::vector<navvis_recon::BinaryOcclusionRayDiagnostic> diagnostics;
            const auto predicted = navvis_recon::classifyBinaryOcclusionRays(
                rays, helper, {}, &diagnostics);
            std::size_t status_exact = 0U;
            std::size_t keep_exact = 0U;
            std::size_t expected_kept = 0U;
            std::size_t predicted_kept = 0U;
            std::size_t confusion[7][7]{};
            std::vector<std::size_t> mismatch_indices;
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                const auto expected_status = std::min<std::uint8_t>(expected[index], 6U);
                const auto predicted_status = std::min<std::uint8_t>(predicted[index], 6U);
                ++confusion[expected_status][predicted_status];
                status_exact += expected_status == predicted_status ? 1U : 0U;
                const bool expected_keep = expected_status >= 1U && expected_status <= 3U;
                const bool predicted_keep = predicted_status >= 1U && predicted_status <= 3U;
                expected_kept += expected_keep ? 1U : 0U;
                predicted_kept += predicted_keep ? 1U : 0U;
                keep_exact += expected_keep == predicted_keep ? 1U : 0U;
                if (expected_status != predicted_status) mismatch_indices.push_back(index);
            }
            std::cout << "{\"stage\":\"occlusion-captured-helper\",\"input\":"
                      << rays.size() << ",\"helper\":" << helper.size()
                      << ",\"expected_kept\":" << expected_kept
                      << ",\"predicted_kept\":" << predicted_kept
                      << ",\"status_exact\":" << status_exact
                      << ",\"keep_mask_exact\":" << keep_exact
                      << ",\"confusion\":[";
            for (std::size_t expected_status = 0U; expected_status < 7U; ++expected_status) {
                if (expected_status != 0U) std::cout << ',';
                std::cout << '[';
                for (std::size_t predicted_status = 0U; predicted_status < 7U;
                     ++predicted_status) {
                    if (predicted_status != 0U) std::cout << ',';
                    std::cout << confusion[expected_status][predicted_status];
                }
                std::cout << ']';
            }
            std::cout << "],\"mismatches\":[";
            for (std::size_t index = 0U; index < mismatch_indices.size(); ++index) {
                if (index != 0U) std::cout << ',';
                const std::size_t point = mismatch_indices[index];
                std::cout << '[' << point << ','
                          << static_cast<unsigned>(expected[point]) << ','
                          << static_cast<unsigned>(predicted[point]) << ",[";
                for (std::size_t candidate = 0U;
                     candidate < diagnostics[point].first_leaf_candidates.size();
                     ++candidate) {
                    if (candidate != 0U) std::cout << ',';
                    std::cout << diagnostics[point].first_leaf_candidates[candidate];
                }
                std::cout << "]," << diagnostics[point].endpoint_neighbor << ','
                          << (diagnostics[point].pair_consistent ? 1 : 0) << ']';
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--occlusion-indices" || mode == "--occlusion-input-indices") {
            if (argc != 5) {
                throw std::invalid_argument(
                    "occlusion indices mode requires raw, helper and index captures");
            }
            const auto all_rays = readSurfaceInputs(argv[2]);
            const auto helper = mode == "--occlusion-indices"
                                    ? readSurfacePoints(argv[3])
                                    : navvis_recon::applyBinaryOcclusionHelperSurfaceFromInput(
                                          readSurfaceInputs(argv[3]));
            std::vector<std::size_t> source_indices;
            std::string requested = argv[4];
            std::size_t begin = 0U;
            while (begin < requested.size()) {
                const std::size_t end = requested.find(',', begin);
                source_indices.push_back(static_cast<std::size_t>(std::stoull(
                    requested.substr(begin, end == std::string::npos
                                                ? std::string::npos
                                                : end - begin))));
                if (end == std::string::npos) {
                    break;
                }
                begin = end + 1U;
            }
            std::vector<navvis_recon::BinarySurfaceInput> rays;
            rays.reserve(source_indices.size());
            for (const auto index : source_indices) {
                if (index >= all_rays.size()) {
                    throw std::out_of_range("occlusion diagnostic index is outside raw input");
                }
                rays.push_back(all_rays[index]);
            }
            std::vector<navvis_recon::BinaryOcclusionRayDiagnostic> diagnostics;
            const auto statuses = navvis_recon::classifyBinaryOcclusionRays(
                rays, helper, {}, &diagnostics);
            std::cout << "{\"stage\":\"occlusion-indices\",\"raw\":"
                      << all_rays.size() << ",\"helper\":" << helper.size()
                      << ",\"results\":[";
            for (std::size_t index = 0U; index < rays.size(); ++index) {
                if (index != 0U) {
                    std::cout << ',';
                }
                std::cout << "{\"index\":" << source_indices[index]
                          << ",\"status\":" << static_cast<unsigned>(statuses[index])
                          << ",\"first_non_miss\":"
                          << static_cast<unsigned>(diagnostics[index].first_non_miss_primitive)
                          << ",\"endpoint_neighbor\":"
                          << diagnostics[index].endpoint_neighbor
                          << ",\"pair_consistent\":"
                          << (diagnostics[index].pair_consistent ? 1 : 0)
                          << ",\"candidates\":[";
                for (std::size_t candidate = 0U;
                     candidate < diagnostics[index].first_leaf_candidates.size();
                     ++candidate) {
                    if (candidate != 0U) {
                        std::cout << ',';
                    }
                    std::cout << diagnostics[index].first_leaf_candidates[candidate];
                }
                std::cout << "]}";
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--occlusion") {
            if (argc != 3) {
                throw std::invalid_argument("--occlusion requires one raw/status capture");
            }
            std::vector<std::uint8_t> expected;
            const auto rays = readRawStatusInputs(argv[2], &expected);
            std::vector<std::uint8_t> predicted;
            navvis_recon::BinaryOcclusionStageCounts stage_counts;
            const auto kept = navvis_recon::applyBinaryOcclusionCleaning(
                rays, &predicted, {}, {}, &stage_counts);
            std::size_t status_exact = 0U;
            std::size_t keep_exact = 0U;
            std::size_t expected_kept = 0U;
            std::size_t predicted_kept = 0U;
            std::size_t confusion[7][7]{};
            for (std::size_t index = 0U; index < expected.size(); ++index) {
                const auto expected_status = std::min<std::uint8_t>(expected[index], 6U);
                const auto predicted_status = std::min<std::uint8_t>(predicted[index], 6U);
                ++confusion[expected_status][predicted_status];
                status_exact += expected_status == predicted_status ? 1U : 0U;
                const bool expected_keep = expected_status >= 1U && expected_status <= 3U;
                const bool predicted_keep = predicted_status >= 1U && predicted_status <= 3U;
                expected_kept += expected_keep ? 1U : 0U;
                predicted_kept += predicted_keep ? 1U : 0U;
                keep_exact += expected_keep == predicted_keep ? 1U : 0U;
            }
            std::cout << "{\"stage\":\"occlusion\",\"input\":" << rays.size()
                      << ",\"expected_kept\":" << expected_kept
                      << ",\"predicted_kept\":" << predicted_kept
                      << ",\"returned\":" << kept.size()
                      << ",\"helper_input\":" << stage_counts.helper_input
                      << ",\"helper_valid\":" << stage_counts.helper_valid
                      << ",\"helper_initial_voxels\":"
                      << stage_counts.helper_initial_voxels
                      << ",\"helper_output\":" << stage_counts.helper_output
                      << ",\"status_exact\":" << status_exact
                      << ",\"keep_mask_exact\":" << keep_exact
                      << ",\"confusion\":[";
            for (std::size_t expected_status = 0U; expected_status < 7U; ++expected_status) {
                if (expected_status != 0U) std::cout << ',';
                std::cout << '[';
                for (std::size_t predicted_status = 0U; predicted_status < 7U;
                     ++predicted_status) {
                    if (predicted_status != 0U) std::cout << ',';
                    std::cout << confusion[expected_status][predicted_status];
                }
                std::cout << ']';
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--occlusion-main-input") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--occlusion-main-input requires raw/status and expected captures");
            }
            const auto rays = readRawStatusInputs(argv[2]);
            const auto expected = readSurfaceInputs(argv[3]);
            const auto kept = navvis_recon::applyBinaryOcclusionCleaning(rays);
            const auto output =
                navvis_recon::applyBinaryOcclusionHelperInputAggregation(kept, 0.01F);
            const std::size_t common = std::min(expected.size(), output.size());
            std::size_t xyz_bit_exact = 0U;
            std::size_t origin_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            for (std::size_t index = 0U; index < common; ++index) {
                xyz_bit_exact += std::memcmp(
                    expected[index].xyz.data(), output[index].xyz.data(), 12U) == 0;
                origin_bit_exact += std::memcmp(
                    expected[index].origin.data(), output[index].origin.data(), 12U) == 0;
                intensity_bit_exact += std::memcmp(
                    &expected[index].intensity, &output[index].intensity, 4U) == 0;
                weight_bit_exact += std::memcmp(
                    &expected[index].weight, &output[index].weight, 4U) == 0;
            }
            std::cout << "{\"stage\":\"occlusion-main-input\",\"raw\":"
                      << rays.size() << ",\"kept\":" << kept.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"origin_bit_exact\":" << origin_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact << "}\n";
            return 0;
        }
        if (mode == "--normal") {
            if (argc != 4) {
                throw std::invalid_argument("--normal requires input and expected captures");
            }
            const auto normal_input = readSurfaceInputs(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            const auto output =
                navvis_recon::applyBinaryMultiScaleNormalEstimation(normal_input);
            std::size_t semantic_record_bit_exact = 0U;
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t curvature_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            std::size_t predicted_zero = 0U;
            std::size_t expected_zero = 0U;
            std::size_t zero_match = 0U;
            std::vector<double> angles;
            std::vector<std::size_t> mismatch_indices;
            double maximum_angle = 0.0;
            std::size_t maximum_angle_index = 0U;
            const std::size_t common = std::min(output.size(), expected.size());
            angles.reserve(common);
            for (std::size_t index = 0U; index < common; ++index) {
                const bool xyz_is_bit_exact =
                    std::memcmp(output[index].xyz.data(), expected[index].xyz.data(),
                                3U * sizeof(float)) == 0;
                const bool normal_is_bit_exact =
                    std::memcmp(output[index].normal.data(), expected[index].normal.data(),
                                3U * sizeof(float)) == 0;
                const bool intensity_is_bit_exact =
                    std::memcmp(&output[index].intensity, &expected[index].intensity,
                                sizeof(float)) == 0;
                const bool curvature_is_bit_exact =
                    std::memcmp(&output[index].curvature, &expected[index].curvature,
                                sizeof(float)) == 0;
                const bool weight_is_bit_exact =
                    std::memcmp(&output[index].weight, &expected[index].weight,
                                sizeof(float)) == 0;
                const bool semantic_record_is_bit_exact =
                    xyz_is_bit_exact && normal_is_bit_exact && intensity_is_bit_exact &&
                    curvature_is_bit_exact && weight_is_bit_exact;
                xyz_bit_exact += xyz_is_bit_exact ? 1U : 0U;
                if (normal_is_bit_exact) {
                    ++normal_bit_exact;
                } else {
                    mismatch_indices.push_back(index);
                }
                intensity_bit_exact += intensity_is_bit_exact ? 1U : 0U;
                curvature_bit_exact += curvature_is_bit_exact ? 1U : 0U;
                weight_bit_exact += weight_is_bit_exact ? 1U : 0U;
                semantic_record_bit_exact += semantic_record_is_bit_exact ? 1U : 0U;
                const bool output_is_zero = output[index].normal.squaredNorm() == 0.0F;
                const bool expected_is_zero = expected[index].normal.squaredNorm() == 0.0F;
                predicted_zero += output_is_zero ? 1U : 0U;
                expected_zero += expected_is_zero ? 1U : 0U;
                zero_match += output_is_zero == expected_is_zero ? 1U : 0U;
                if (!output_is_zero && !expected_is_zero) {
                    // Identical float32 vectors have exactly zero angular error.  Computing
                    // dot(n, n) / (norm(n) * norm(n)) can nevertheless round below one in
                    // double and report a fictitious microdegree error, which obscures the
                    // stronger bit-exact acceptance result.
                    double angle = 0.0;
                    if (!normal_is_bit_exact) {
                        const auto output_normal = output[index].normal.cast<double>();
                        const auto expected_normal = expected[index].normal.cast<double>();
                        const double denominator = output_normal.norm() * expected_normal.norm();
                        const double cosine = std::clamp(
                            output_normal.dot(expected_normal) / denominator,
                            -1.0, 1.0);
                        angle = 180.0 / 3.14159265358979323846 * std::acos(cosine);
                    }
                    angles.push_back(angle);
                    if (angle > maximum_angle) {
                        maximum_angle = angle;
                        maximum_angle_index = index;
                    }
                }
            }
            std::sort(angles.begin(), angles.end());
            const auto percentile = [&angles](double fraction) {
                if (angles.empty()) return 0.0;
                const std::size_t index = std::min(
                    angles.size() - 1U,
                    static_cast<std::size_t>(fraction * static_cast<double>(angles.size() - 1U)));
                return angles[index];
            };
            std::cout << "{\"stage\":\"normal\",\"input\":" << normal_input.size()
                      << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"semantic_record_bit_exact\":" << semantic_record_bit_exact
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"curvature_bit_exact\":" << curvature_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"predicted_zero\":" << predicted_zero
                      << ",\"expected_zero\":" << expected_zero
                      << ",\"zero_mask_match\":" << zero_match
                      << ",\"angle_p50_degrees\":" << percentile(0.50)
                      << ",\"angle_p95_degrees\":" << percentile(0.95)
                      << ",\"angle_max_degrees\":" << percentile(1.0)
                      << ",\"angle_max_index\":" << maximum_angle_index
                      << ",\"mismatch_samples\":[";
            const std::size_t sample_count = std::min<std::size_t>(mismatch_indices.size(), 24U);
            for (std::size_t sample = 0U; sample < sample_count; ++sample) {
                if (sample != 0U) std::cout << ',';
                const std::size_t index = mismatch_indices[sample];
                std::uint32_t expected_bits[3]{};
                std::uint32_t output_bits[3]{};
                std::memcpy(expected_bits, expected[index].normal.data(), 12U);
                std::memcpy(output_bits, output[index].normal.data(), 12U);
                std::cout << "{\"index\":" << index << ",\"expected\":["
                          << expected_bits[0] << ',' << expected_bits[1] << ','
                          << expected_bits[2] << "],\"output\":["
                          << output_bits[0] << ',' << output_bits[1] << ','
                          << output_bits[2] << "]}";
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--voxel-compare") {
            if (argc != 4) {
                throw std::invalid_argument(
                    "--voxel-compare requires input and expected captures");
            }
            const auto voxel_input = readSurfacePoints(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            const auto output =
                navvis_recon::applyBinaryOutputVoxelAggregation(voxel_input);
            using Key = std::array<int, 3>;
            const auto voxel_key = [](const auto& point) {
                constexpr double inverse = 100.00000223517424;
                return Key{
                    static_cast<int>(std::floor(double(point.xyz.x()) * inverse)),
                    static_cast<int>(std::floor(double(point.xyz.y()) * inverse)),
                    static_cast<int>(std::floor(double(point.xyz.z()) * inverse))};
            };
            std::map<Key, const navvis_recon::BinarySurfacePoint*> expected_by_key;
            std::map<Key, const navvis_recon::BinarySurfacePoint*> output_by_key;
            for (const auto& point : expected) expected_by_key[voxel_key(point)] = &point;
            for (const auto& point : output) output_by_key[voxel_key(point)] = &point;
            std::size_t common = 0U;
            std::size_t record_bit_exact = 0U;
            std::size_t xyz_bit_exact = 0U;
            std::size_t normal_bit_exact = 0U;
            std::size_t intensity_bit_exact = 0U;
            std::size_t curvature_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            double xyz_sum = 0.0;
            std::size_t mismatch_samples = 0U;
            for (const auto& [key, expected_point] : expected_by_key) {
                const auto found = output_by_key.find(key);
                if (found == output_by_key.end()) continue;
                ++common;
                const bool xyz_exact = std::memcmp(
                    expected_point->xyz.data(), found->second->xyz.data(), 12U) == 0;
                const bool normal_exact = std::memcmp(
                    expected_point->normal.data(), found->second->normal.data(), 12U) == 0;
                const bool intensity_exact = std::memcmp(
                    &expected_point->intensity, &found->second->intensity, 4U) == 0;
                const bool curvature_exact = std::memcmp(
                    &expected_point->curvature, &found->second->curvature, 4U) == 0;
                const bool weight_exact = std::memcmp(
                    &expected_point->weight, &found->second->weight, 4U) == 0;
                xyz_bit_exact += xyz_exact;
                normal_bit_exact += normal_exact;
                intensity_bit_exact += intensity_exact;
                curvature_bit_exact += curvature_exact;
                weight_bit_exact += weight_exact;
                record_bit_exact += xyz_exact && normal_exact && intensity_exact &&
                                    curvature_exact && weight_exact;
                xyz_sum += (expected_point->xyz - found->second->xyz).norm();
                if ((!xyz_exact || !normal_exact || !intensity_exact || !curvature_exact ||
                     !weight_exact) && mismatch_samples < 12U) {
                    std::uint32_t xyz_output_bits[3]{};
                    std::uint32_t xyz_expected_bits[3]{};
                    std::uint32_t normal_output_bits[3]{};
                    std::uint32_t normal_expected_bits[3]{};
                    std::memcpy(xyz_output_bits, found->second->xyz.data(), 12U);
                    std::memcpy(xyz_expected_bits, expected_point->xyz.data(), 12U);
                    std::memcpy(normal_output_bits, found->second->normal.data(), 12U);
                    std::memcpy(normal_expected_bits, expected_point->normal.data(), 12U);
                    std::cerr << "voxel mismatch key=" << key[0] << ',' << key[1] << ','
                              << key[2] << " xyz_bits_out=" << xyz_output_bits[0] << ','
                              << xyz_output_bits[1] << ',' << xyz_output_bits[2]
                              << " xyz_bits_expected=" << xyz_expected_bits[0] << ','
                              << xyz_expected_bits[1] << ',' << xyz_expected_bits[2]
                              << " normal_bits_out=" << normal_output_bits[0] << ','
                              << normal_output_bits[1] << ',' << normal_output_bits[2]
                              << " normal_bits_expected=" << normal_expected_bits[0] << ','
                              << normal_expected_bits[1] << ',' << normal_expected_bits[2]
                              << " normal_exact=" << normal_exact
                              << " intensity_exact=" << intensity_exact
                              << " curvature_exact=" << curvature_exact
                              << " weight_exact=" << weight_exact << '\n';
                    ++mismatch_samples;
                }
            }
            std::cout << "{\"stage\":\"voxel-compare\",\"input\":"
                      << voxel_input.size() << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"expected_keys\":" << expected_by_key.size()
                      << ",\"output_keys\":" << output_by_key.size()
                      << ",\"common_keys\":" << common
                      << ",\"record_bit_exact\":" << record_bit_exact
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"normal_bit_exact\":" << normal_bit_exact
                      << ",\"intensity_bit_exact\":" << intensity_bit_exact
                      << ",\"curvature_bit_exact\":" << curvature_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"xyz_mean\":" << (common ? xyz_sum / common : 0.0)
                      << ",\"missing_keys\":[";
            bool first = true;
            for (const auto& [key, point] : expected_by_key) {
                (void)point;
                if (output_by_key.find(key) != output_by_key.end()) continue;
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << key[0] << ',' << key[1] << ',' << key[2] << ']';
            }
            std::cout << "],\"extra_keys\":[";
            first = true;
            for (const auto& [key, point] : output_by_key) {
                (void)point;
                if (expected_by_key.find(key) != expected_by_key.end()) continue;
                if (!first) std::cout << ',';
                first = false;
                std::cout << '[' << key[0] << ',' << key[1] << ',' << key[2] << ']';
            }
            std::cout << "]}\n";
            return 0;
        }
        if (mode == "--selection") {
            if (argc != 4) {
                throw std::invalid_argument("--selection requires input and expected captures");
            }
            const auto selection_input = readSurfacePoints(argv[2]);
            const auto expected = readSurfacePoints(argv[3]);
            const auto output = navvis_recon::applyBinarySurfacePointSelection(selection_input);
            std::size_t xyz_bit_exact = 0U;
            std::size_t weight_bit_exact = 0U;
            float maximum_xyz_error = 0.0F;
            const std::size_t common = std::min(output.size(), expected.size());
            for (std::size_t index = 0U; index < common; ++index) {
                if (std::memcmp(output[index].xyz.data(), expected[index].xyz.data(),
                                3U * sizeof(float)) == 0) {
                    ++xyz_bit_exact;
                }
                if (std::memcmp(&output[index].weight, &expected[index].weight,
                                sizeof(float)) == 0) {
                    ++weight_bit_exact;
                }
                maximum_xyz_error = std::max(
                    maximum_xyz_error,
                    (output[index].xyz - expected[index].xyz).cwiseAbs().maxCoeff());
            }
            std::cout << "{\"stage\":\"selection\",\"input\":"
                      << selection_input.size() << ",\"expected\":" << expected.size()
                      << ",\"output\":" << output.size()
                      << ",\"xyz_bit_exact\":" << xyz_bit_exact
                      << ",\"weight_bit_exact\":" << weight_bit_exact
                      << ",\"maximum_xyz_error\":" << maximum_xyz_error << "}\n";
            return 0;
        }
        std::ifstream input(argv[argc - 1], std::ios::binary);
        if (!input) throw std::runtime_error("cannot open capture");
        std::vector<navvis_recon::BinarySurfaceInput> points;
        CapturedPoint48 record{};
        std::vector<navvis_recon::BinarySurfacePoint> surface_points;
        while (input.read(reinterpret_cast<char*>(&record), sizeof(record))) {
            if (mode == "--full") {
                points.push_back({
                    navvis_recon::Vec3f(record.field[4], record.field[5], record.field[6]),
                    navvis_recon::Vec3f(record.field[0], record.field[1], record.field[2]),
                    record.field[8], record.field[9]});
            } else {
                surface_points.push_back({
                    navvis_recon::Vec3f(record.field[4], record.field[5], record.field[6]),
                    navvis_recon::Vec3f(record.field[0], record.field[1], record.field[2]),
                    record.field[8], record.field[9], record.field[10]});
            }
        }
        if (mode != "--full") {
            std::vector<navvis_recon::BinarySurfacePoint> output;
            if (mode == "--voxel") {
                output = navvis_recon::applyBinaryOutputVoxelAggregation(surface_points);
            } else if (mode == "--density") {
                output = navvis_recon::applyBinaryDensityFilter(surface_points);
            } else if (mode == "--sor") {
                output = navvis_recon::applyBinaryAdaptiveSor(surface_points);
            } else {
                throw std::invalid_argument("unknown capture mode " + mode);
            }
            std::cout << "{\"stage\":\"" << mode.substr(2)
                      << "\",\"input\":" << surface_points.size()
                      << ",\"output\":" << output.size() << "}\n";
            return 0;
        }
        navvis_recon::BinarySurfaceStageCounts counts;
        const auto output = navvis_recon::runBinarySurfacePipeline(
            points, navvis_recon::BinarySurfaceOptions{}, &counts);
        std::cout << "{\"input\":" << counts.input
                  << ",\"normals\":" << counts.normals
                  << ",\"selected\":" << counts.selected
                  << ",\"valid\":" << counts.valid
                  << ",\"output_voxels\":" << counts.output_voxels
                  << ",\"density\":" << counts.density
                  << ",\"adaptive_sor\":" << counts.adaptive_sor
                  << ",\"post\":" << counts.post
                  << ",\"returned\":" << output.size() << "}\n";
        return 0;
    } catch (const std::exception& error) {
        std::cerr << "surface_capture_acceptance: " << error.what() << '\n';
        return 1;
    }
}
