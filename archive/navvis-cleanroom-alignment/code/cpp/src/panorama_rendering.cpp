#include "navvis_recon/panorama_rendering.hpp"

#define PCL_NO_PRECOMPILE
#include <pcl/octree/octree_search.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <Eigen/SVD>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <stdexcept>

namespace navvis_recon {
namespace {

std::vector<cv::Mat> gaussianPyramid(const cv::Mat& image, int requested_levels) {
    std::vector<cv::Mat> pyramid{image.clone()};
    for (int level = 1; level < requested_levels; ++level) {
        if (std::min(pyramid.back().rows, pyramid.back().cols) <= 2) {
            break;
        }
        cv::Mat next;
        cv::pyrDown(pyramid.back(), next);
        pyramid.push_back(next);
    }
    return pyramid;
}

std::vector<cv::Mat> laplacianPyramid(const cv::Mat& image, int levels) {
    auto gaussian = gaussianPyramid(image, levels);
    std::vector<cv::Mat> laplacian;
    for (std::size_t level = 0; level + 1 < gaussian.size(); ++level) {
        cv::Mat expanded;
        cv::pyrUp(gaussian[level + 1], expanded, gaussian[level].size());
        laplacian.push_back(gaussian[level] - expanded);
    }
    laplacian.push_back(gaussian.back());
    return laplacian;
}

cv::Mat1b unionMask(const std::vector<cv::Mat1f>& masks) {
    cv::Mat1b result(masks.front().size(), static_cast<std::uint8_t>(0));
    for (const auto& mask : masks) {
        result |= mask > 0.0F;
    }
    return result;
}

struct DepthLinearSystem {
    int width;
    int height;
    Eigen::VectorXd measured;
    std::vector<std::uint8_t> valid;
    Eigen::VectorXd inverse_diagonal;
    double ray_weight;
};

DepthLinearSystem makeDepthLinearSystem(
    const cv::Mat1d& measured, const cv::Mat1b& valid, double ray_weight) {
    const Eigen::Index count = static_cast<Eigen::Index>(measured.total());
    DepthLinearSystem system{
        measured.cols,
        measured.rows,
        Eigen::Map<const Eigen::VectorXd>(measured.ptr<double>(), count),
        std::vector<std::uint8_t>(valid.datastart, valid.dataend),
        Eigen::VectorXd::Zero(count),
        ray_weight,
    };
    for (int row = 0; row < measured.rows; ++row) {
        for (int column = 0; column < measured.cols; ++column) {
            const Eigen::Index index =
                static_cast<Eigen::Index>(row) * measured.cols + column;
            // Match the installed scalar instruction order.  In particular,
            // the data diagonal is accumulated before the vertical degree
            // and the constant horizontal degree.
            double diagonal = 0.0;
            if (system.valid[static_cast<std::size_t>(index)] != 0U) {
                diagonal = ray_weight;
                diagonal *= 0.05;
                diagonal *= system.measured[index];
                diagonal += 0.0;
            }
            if (row != 0) {
                diagonal += 1.0;
            }
            if (row + 1 < measured.rows) {
                diagonal += 1.0;
            }
            diagonal += 2.0;
            system.inverse_diagonal[index] =
                diagonal > 1.0e-6 ? 1.0 / diagonal : 1.0;
        }
    }
    return system;
}

Eigen::VectorXd negativeDepthGradient(
    const DepthLinearSystem& system, const Eigen::VectorXd& estimate) {
    Eigen::VectorXd residual(estimate.size());
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const Eigen::Index index =
                static_cast<Eigen::Index>(row) * system.width + column;
            const double current = estimate[index];
            double non_horizontal = 0.0;
            if (system.valid[static_cast<std::size_t>(index)] != 0U) {
                non_horizontal = system.ray_weight;
                non_horizontal *= 0.05;
                non_horizontal *= system.measured[index];
                non_horizontal *= current - system.measured[index];
                non_horizontal += 0.0;
            }
            if (row != 0) {
                non_horizontal += current - estimate[index - system.width];
            }
            if (row + 1 != system.height) {
                non_horizontal += current + estimate[index + system.width] * -1.0;
            }
            const int right_column =
                column + 1 == system.width ? 0 : column + 1;
            const int left_column = column == 0 ? system.width - 1 : column - 1;
            double horizontal = current + current;
            horizontal -= estimate[static_cast<Eigen::Index>(row) * system.width +
                                   right_column];
            horizontal -= estimate[static_cast<Eigen::Index>(row) * system.width +
                                   left_column];
            horizontal += non_horizontal;
            residual[index] = -horizontal;
        }
    }
    return residual;
}

Eigen::VectorXd applyDepthHessian(
    const DepthLinearSystem& system, const Eigen::VectorXd& input) {
    Eigen::VectorXd output(input.size());
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const Eigen::Index index =
                static_cast<Eigen::Index>(row) * system.width + column;
            const double current = input[index];
            double non_horizontal = 0.0;
            if (system.valid[static_cast<std::size_t>(index)] != 0U) {
                non_horizontal = system.ray_weight;
                non_horizontal *= 0.05;
                non_horizontal *= system.measured[index];
                non_horizontal *= current;
                non_horizontal += 0.0;
            }
            if (row != 0) {
                non_horizontal += current - input[index - system.width];
            }
            if (row + 1 != system.height) {
                non_horizontal += current + input[index + system.width] * -1.0;
            }
            const int right_column =
                column + 1 == system.width ? 0 : column + 1;
            const int left_column = column == 0 ? system.width - 1 : column - 1;
            double horizontal = current + current;
            // The installed worker subtracts the right longitude neighbour
            // before the left one.  Reversing these algebraically equivalent
            // operations changes several hundred PCG values by a few ULPs.
            horizontal -= input[static_cast<Eigen::Index>(row) * system.width +
                                right_column];
            horizontal -= input[static_cast<Eigen::Index>(row) * system.width +
                                left_column];
            output[index] = horizontal + non_horizontal;
        }
    }
    return output;
}

double depthObjective(
    const DepthLinearSystem& system, const Eigen::VectorXd& estimate) {
    double result = 0.0;
    for (int row = 0; row < system.height; ++row) {
        for (int column = 0; column < system.width; ++column) {
            const Eigen::Index index =
                static_cast<Eigen::Index>(row) * system.width + column;
            const double current = estimate[index];
            if (system.valid[static_cast<std::size_t>(index)] != 0U) {
                const double difference = current - system.measured[index];
                result += 0.5 * system.ray_weight * 0.05 *
                          system.measured[index] * difference * difference;
            }
            const int right_column =
                column + 1 == system.width ? 0 : column + 1;
            const double horizontal_difference =
                current - estimate[static_cast<Eigen::Index>(row) * system.width +
                                   right_column];
            result += 0.5 * horizontal_difference * horizontal_difference;
            if (row + 1 != system.height) {
                const double vertical_difference =
                    current - estimate[index + system.width];
                result += 0.5 * vertical_difference * vertical_difference;
            }
        }
    }
    return result;
}

cv::Mat1d solveDepthLevel(
    const cv::Mat1d& measured, const cv::Mat1b& valid,
    const cv::Mat1d& initial, double ray_weight, int maximum_iterations,
    bool zero_initial) {
    DepthLinearSystem system =
        makeDepthLinearSystem(measured, valid, ray_weight);
    const Eigen::Index count = static_cast<Eigen::Index>(initial.total());
    Eigen::VectorXd estimate =
        Eigen::Map<const Eigen::VectorXd>(initial.ptr<double>(), count);
    if (zero_initial) {
        estimate.setZero();
    }
    double previous_objective = depthObjective(system, estimate);
    int completed_outer_iterations = 0;
    int completed_inner_iterations = 0;
    double final_residual_squared = 0.0;
    for (int outer = 0; outer < 100; ++outer) {
        Eigen::VectorXd residual = negativeDepthGradient(system, estimate);
        final_residual_squared = residual.squaredNorm();
        if (1.0e-15 > final_residual_squared) {
            break;
        }
        Eigen::VectorXd preconditioned =
            residual.cwiseProduct(system.inverse_diagonal);
        Eigen::VectorXd direction = preconditioned;
        double residual_product = preconditioned.dot(residual);
        Eigen::VectorXd delta = Eigen::VectorXd::Zero(estimate.size());
        for (int inner = 0; inner < maximum_iterations; ++inner) {
            const Eigen::VectorXd hessian_times_direction =
                applyDepthHessian(system, direction);
            const double denominator = hessian_times_direction.dot(direction);
            const double step = residual_product / denominator;
            delta += step * direction;
            residual -= step * hessian_times_direction;
            ++completed_inner_iterations;
            final_residual_squared = residual.squaredNorm();
            if (1.0e-15 > final_residual_squared) {
                break;
            }
            preconditioned = residual.cwiseProduct(system.inverse_diagonal);
            const double next_product = preconditioned.dot(residual);
            const double beta = next_product / residual_product;
            direction = preconditioned + beta * direction;
            residual_product = next_product;
        }
        estimate += delta;
        ++completed_outer_iterations;
        const double next_objective = depthObjective(system, estimate);
        double relative_change = std::abs(next_objective - previous_objective);
        if (previous_objective > 0.0) {
            relative_change /= previous_objective;
        }
        if (1.0e-6 > relative_change) {
            break;
        }
        previous_objective = next_objective;
    }
    if (std::getenv("NAVVIS_RECON_DEPTH_TRACE") != nullptr) {
        std::cerr << "Depth PCG " << measured.cols << 'x' << measured.rows
                  << ": outer=" << completed_outer_iterations
                  << ", inner=" << completed_inner_iterations
                  << ", r2=" << final_residual_squared << '\n';
    }
    cv::Mat1d result(measured.size());
    Eigen::Map<Eigen::VectorXd>(result.ptr<double>(), count) = estimate;
    return result;
}

void appendReducedDepthLevel(
    std::vector<cv::Mat1d>& measured_pyramid,
    std::vector<cv::Mat1b>& mask_pyramid) {
    const cv::Mat1d& source = measured_pyramid.back();
    const cv::Mat1b& source_mask = mask_pyramid.back();
    cv::Mat1d reduced = cv::Mat1d::zeros(source.rows / 2, source.cols / 2);

    for (int row = 0; row < reduced.rows; ++row) {
        for (int column = 0; column < reduced.cols; ++column) {
            double sum = 0.0;
            int count = 0;
            for (int source_row = 2 * row;
                 source_row < std::min(source.rows, 2 * row + 2); ++source_row) {
                for (int source_column = 2 * column;
                     source_column < std::min(source.cols, 2 * column + 2);
                     ++source_column) {
                    if (source_mask(source_row, source_column) != 0U) {
                        sum += source(source_row, source_column);
                        ++count;
                    }
                }
            }
            if (count != 0) {
                // The installed renderer materializes the reciprocal before
                // multiplying the accumulated valid samples.  This differs
                // from a direct division by one ULP when exactly three of the
                // four source pixels are valid, and that difference propagates
                // through the finer PCG levels.
                reduced(row, column) =
                    sum * (1.0 / static_cast<double>(count));
            }
        }
    }
    measured_pyramid.push_back(std::move(reduced));
    mask_pyramid.push_back(measured_pyramid.back() > 0.0);
}

}  // namespace

cv::Mat1d GaussNewtonDepthMapOptimizer::optimizeDouble(
    const cv::Mat1f& measured_depth, const cv::Mat1b& valid_mask,
    double ray_weight, int iterations) {
    if (measured_depth.empty() || measured_depth.size() != valid_mask.size() ||
        ray_weight <= 0.0 || iterations <= 0 || cv::countNonZero(valid_mask) == 0) {
        throw std::invalid_argument("Depth optimization needs valid measurements and weights");
    }

    cv::Mat1d finest_depth;
    measured_depth.convertTo(finest_depth, CV_64F);
    cv::Mat1b finest_mask = (valid_mask > 0) & (measured_depth > 0.0F);
    finest_depth.setTo(0.0, finest_mask == 0);

    const int minimum_dimension =
        std::min(measured_depth.rows, measured_depth.cols);
    const int level_count = std::min(
        4, std::max(1, static_cast<int>(std::lround(std::log2(minimum_dimension)))));
    std::vector<cv::Mat1d> measured_pyramid{std::move(finest_depth)};
    std::vector<cv::Mat1b> mask_pyramid{std::move(finest_mask)};
    for (int level = 1; level < level_count; ++level) {
        if (measured_pyramid.back().rows < 2 || measured_pyramid.back().cols < 2) {
            break;
        }
        appendReducedDepthLevel(measured_pyramid, mask_pyramid);
    }

    cv::Mat1d estimate = measured_pyramid.back().clone();
    for (int level = static_cast<int>(measured_pyramid.size()) - 1;
         level >= 0; --level) {
        const cv::Mat1d& measured = measured_pyramid[static_cast<std::size_t>(level)];
        const cv::Mat1b& mask = mask_pyramid[static_cast<std::size_t>(level)];
        if (estimate.size() != measured.size()) {
            cv::Mat1d upsampled;
            cv::resize(estimate, upsampled, measured.size(), 0.0, 0.0, cv::INTER_LINEAR);
            estimate = measured.clone();
            upsampled.copyTo(estimate, mask == 0);
        }
        estimate = solveDepthLevel(
            measured, mask, estimate, ray_weight, iterations,
            level == static_cast<int>(measured_pyramid.size()) - 1);
    }
    return estimate;
}

cv::Mat1f GaussNewtonDepthMapOptimizer::optimize(
    const cv::Mat1f& measured_depth, const cv::Mat1b& valid_mask,
    float ray_weight, int iterations) {
    cv::Mat1f result;
    optimizeDouble(measured_depth, valid_mask, ray_weight, iterations)
        .convertTo(result, CV_32F);
    return result;
}

std::vector<cv::Vec3f> ExposureCompensatorSoftConstraint::estimateGains(
    const std::vector<cv::Mat3b>& images, const std::vector<cv::Mat1b>& masks) {
    const std::size_t count = images.size();
    if (count == 0U) {
        return {};
    }
    if (masks.size() != count) {
        throw std::invalid_argument("Exposure images and masks must have equal size");
    }
    for (std::size_t index = 0; index < count; ++index) {
        if (images[index].empty() || masks[index].empty() ||
            images[index].size() != masks[index].size() ||
            images[index].size() != images.front().size()) {
            throw std::invalid_argument(
                "Exposure images and masks must be non-empty and equally sized");
        }
    }

    struct Overlap {
        std::size_t first;
        std::size_t second;
        double root_pixels;
        cv::Scalar first_mean;
        cv::Scalar second_mean;
    };
    std::vector<Overlap> overlaps;
    overlaps.reserve(count > 0U ? count - 1U : 0U);
    for (std::size_t first = 0; first + 1U < count; ++first) {
        const std::size_t second = first + 1U;
        const cv::Mat1b logical_overlap =
            (masks[first] > 0) & (masks[second] > 0);
        const int pixels = cv::countNonZero(logical_overlap);
        if (pixels == 0) {
            continue;
        }
        cv::Mat1b mean_mask;
        cv::bitwise_and(masks[first], masks[second], mean_mask);
        overlaps.push_back({
            first,
            second,
            std::sqrt(static_cast<double>(pixels)),
            cv::mean(images[first], mean_mask),
            cv::mean(images[second], mean_mask),
        });
    }

    std::vector<cv::Vec3f> gains(count, cv::Vec3f(1.0F, 1.0F, 1.0F));
    for (int channel = 0; channel < 3; ++channel) {
        const Eigen::Index rows = static_cast<Eigen::Index>(overlaps.size() + count);
        const Eigen::Index columns = static_cast<Eigen::Index>(count);
        Eigen::MatrixXd system = Eigen::MatrixXd::Zero(rows, columns);
        Eigen::VectorXd target = Eigen::VectorXd::Zero(rows);
        Eigen::VectorXd soft_weights = Eigen::VectorXd::Zero(columns);

        for (std::size_t row = 0; row < overlaps.size(); ++row) {
            const Overlap& overlap = overlaps[row];
            system(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(overlap.first)) =
                overlap.first_mean[channel] * overlap.root_pixels / 15.0;
            system(static_cast<Eigen::Index>(row),
                   static_cast<Eigen::Index>(overlap.second)) =
                -overlap.second_mean[channel] * overlap.root_pixels / 15.0;
            soft_weights[static_cast<Eigen::Index>(overlap.first)] +=
                4.0 * overlap.root_pixels;
            soft_weights[static_cast<Eigen::Index>(overlap.second)] +=
                4.0 * overlap.root_pixels;
        }
        for (std::size_t camera = 0; camera < count; ++camera) {
            double weight = soft_weights[static_cast<Eigen::Index>(camera)];
            if (weight <= 0.0) {
                weight = 3000.0;
            }
            const Eigen::Index row =
                static_cast<Eigen::Index>(overlaps.size() + camera);
            system(row, static_cast<Eigen::Index>(camera)) = weight;
            target[row] = weight;
        }

        Eigen::VectorXd solution = system.jacobiSvd(
            Eigen::ComputeThinU | Eigen::ComputeThinV).solve(target);
        const double squared_norm = solution.squaredNorm();
        if (squared_norm > 0.0) {
            solution *= solution.sum() / squared_norm;
        }
        for (std::size_t camera = 0; camera < count; ++camera) {
            gains[camera][channel] = static_cast<float>(std::clamp(
                solution[static_cast<Eigen::Index>(camera)], 0.6, 2.5));
        }
    }
    return gains;
}

std::vector<cv::Mat1f> SeamMaskPreparer::prepare(
    const std::vector<cv::Mat1f>& masks, PanoramaOptions::SeamFinder finder) {
    if (masks.empty()) {
        return {};
    }
    std::vector<cv::Mat1f> scores;
    scores.reserve(masks.size());
    for (const auto& mask : masks) {
        cv::Mat1f distance;
        cv::distanceTransform(mask > 0.0F, distance, cv::DIST_L2, 3);
        if (finder == PanoramaOptions::SeamFinder::DynamicProgramming) {
            cv::GaussianBlur(distance, distance, cv::Size(1, 9), 0.0);
        } else if (finder == PanoramaOptions::SeamFinder::GraphCut) {
            // A gradient penalty is a deterministic stand-in for OpenCV's
            // detail::GraphCutSeamFinder when the stitching module is absent.
            cv::Mat1f gradient_x;
            cv::Sobel(distance, gradient_x, CV_32F, 1, 0);
            distance -= 0.1F * cv::abs(gradient_x);
        }
        scores.push_back(distance);
    }
    std::vector<cv::Mat1f> result(masks.size());
    for (auto& mask : result) {
        mask = cv::Mat1f::zeros(masks.front().size());
    }
    for (int y = 0; y < masks.front().rows; ++y) {
        for (int x = 0; x < masks.front().cols; ++x) {
            std::size_t owner = 0;
            float best = -1.0F;
            for (std::size_t i = 0; i < scores.size(); ++i) {
                if (masks[i](y, x) > 0.0F && scores[i](y, x) > best) {
                    best = scores[i](y, x);
                    owner = i;
                }
            }
            if (best >= 0.0F) {
                result[owner](y, x) = 1.0F;
            }
        }
    }
    return result;
}

cv::Mat3f MultiBandBlender::blend(
    const std::vector<cv::Mat3f>& images, const std::vector<cv::Mat1f>& masks,
    int levels) {
    if (images.empty() || images.size() != masks.size()) {
        throw std::invalid_argument("images and masks must have equal non-zero size");
    }
    std::vector<std::vector<cv::Mat>> image_pyramids;
    std::vector<std::vector<cv::Mat>> mask_pyramids;
    for (std::size_t i = 0; i < images.size(); ++i) {
        image_pyramids.push_back(laplacianPyramid(images[i], levels));
        mask_pyramids.push_back(gaussianPyramid(masks[i], levels));
    }
    const std::size_t count = image_pyramids.front().size();
    std::vector<cv::Mat3f> blended;
    for (std::size_t level = 0; level < count; ++level) {
        cv::Mat3f numerator = cv::Mat3f::zeros(image_pyramids.front()[level].size());
        cv::Mat1f denominator = cv::Mat1f::zeros(image_pyramids.front()[level].size());
        for (std::size_t image = 0; image < images.size(); ++image) {
            cv::Mat channels[] = {mask_pyramids[image][level], mask_pyramids[image][level],
                                  mask_pyramids[image][level]};
            cv::Mat3f weight;
            cv::merge(channels, 3, weight);
            numerator += image_pyramids[image][level].mul(weight);
            denominator += mask_pyramids[image][level];
        }
        cv::Mat channels[] = {denominator, denominator, denominator};
        cv::Mat3f denominator3;
        cv::merge(channels, 3, denominator3);
        cv::max(denominator3, 1.0e-6, denominator3);
        blended.push_back(numerator / denominator3);
    }
    cv::Mat3f result = blended.back();
    for (std::ptrdiff_t level = static_cast<std::ptrdiff_t>(blended.size()) - 2; level >= 0; --level) {
        cv::Mat3f expanded;
        cv::pyrUp(result, expanded, blended[static_cast<std::size_t>(level)].size());
        result = expanded + blended[static_cast<std::size_t>(level)];
    }
    cv::max(result, 0.0, result);
    cv::min(result, 1.0, result);
    return result;
}

cv::Mat3f PyramidInpainting::fillImage(const cv::Mat3f& image, const cv::Mat1b& valid_mask) {
    const cv::Mat1b missing = valid_mask == 0;
    if (cv::countNonZero(missing) == 0) {
        return image;
    }
    cv::Mat3b input8;
    image.convertTo(input8, CV_8UC3, 255.0);
    cv::Mat3b filled8;
    cv::inpaint(input8, missing, filled8, 3.0, cv::INPAINT_TELEA);
    cv::Mat3f filled;
    filled8.convertTo(filled, CV_32FC3, 1.0 / 255.0);
    return filled;
}

cv::Mat3f FloorFiller::fill(const cv::Mat3f& panorama, const cv::Mat1b& valid_mask) {
    cv::Mat1b floor_mask = valid_mask.clone();
    floor_mask(cv::Rect(0, 0, floor_mask.cols, floor_mask.rows / 2)).setTo(255);
    return PyramidInpainting::fillImage(panorama, floor_mask);
}

cv::Mat1d PanoramaDepthRenderer::render(
    const std::vector<Vec3f>& points_world, const Pose& world_from_head,
    double near_distance, int width, int height) {
    if (points_world.empty() || !(near_distance >= 0.0) || width <= 0 || height <= 0) {
        throw std::invalid_argument("Panorama depth input must be non-empty and sized");
    }
    constexpr double pi = 3.141592653589793238462643383279502884;
    constexpr double voxel_resolution = 0.05;

    auto cloud = pcl::make_shared<pcl::PointCloud<pcl::PointXYZ>>();
    cloud->points.resize(points_world.size());
    cloud->width = static_cast<std::uint32_t>(points_world.size());
    cloud->height = 1U;
    cloud->is_dense = true;
    for (std::size_t index = 0; index < points_world.size(); ++index) {
        const Vec3f& source = points_world[index];
        pcl::PointXYZ& destination = cloud->points[index];
        destination.x = source.x();
        destination.y = source.y();
        destination.z = source.z();
        destination.data[3] = 1.0F;
        cloud->is_dense = cloud->is_dense && pcl::isFinite(destination);
    }

    pcl::octree::OctreePointCloudSearch<pcl::PointXYZ> octree(voxel_resolution);
    octree.setInputCloud(cloud);
    // Calling defineBoundingBox() before insertion is observable: PCL expands
    // the cube around the point bounds differently when it is omitted.
    octree.defineBoundingBox();
    octree.addPointsFromInputCloud();

    const Eigen::Vector3d translation =
        world_from_head.has_double_pose
            ? world_from_head.translation_double
            : world_from_head.translation.cast<double>();
    const Eigen::Quaterniond rotation =
        world_from_head.has_double_pose
            ? world_from_head.rotation_double
            : Eigen::Quaterniond(world_from_head.rotation.cast<double>());
    const pcl::PointXYZ origin(
        static_cast<float>(translation.x()), static_cast<float>(translation.y()),
        static_cast<float>(translation.z()));
    const double near_squared = near_distance * near_distance;
    cv::Mat1f reprojected_depth(height, width, -1.0F);
    std::vector<int> candidates;
    std::size_t projected = 0U;

    const auto panorama_ray = [width, height, pi](int column, int row) {
        const double latitude =
            (0.5 - (static_cast<double>(row) + 0.5) / height) * pi;
        const double longitude =
            ((static_cast<double>(column) + 0.5) / width - 0.5) * (2.0 * pi);
        const double cosine = std::cos(latitude);
        return Eigen::Vector3d(
            cosine * std::cos(longitude), -cosine * std::sin(longitude),
            std::sin(latitude));
    };
    const auto squared_distance = [&origin](const pcl::PointXYZ& point) {
        // Keep the installed scalar/SIMD reduction order: Y+Z, then X.
        const float delta_z = origin.z - point.z;
        const float delta_y = origin.y - point.y;
        const float delta_x = origin.x - point.x;
        const float squared_z = delta_z * delta_z;
        const float squared_y = delta_y * delta_y;
        const float squared_x = delta_x * delta_x;
        const float squared_yz = squared_y + squared_z;
        return squared_x + squared_yz;
    };

    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const Eigen::Vector3d point_world =
                rotation * panorama_ray(column, row) + translation;
            pcl::PointXYZ direction;
            // The binary casts the transformed endpoint to float before
            // subtracting the float ray origin.
            direction.x = static_cast<float>(point_world.x()) - origin.x;
            direction.y = static_cast<float>(point_world.y()) - origin.y;
            direction.z = static_cast<float>(point_world.z()) - origin.z;
            direction.data[3] = 1.0F;
            octree.getIntersectedVoxelIndices(
                origin.getVector3fMap(), direction.getVector3fMap(), candidates);

            int best_index = -1;
            double best_squared = std::numeric_limits<double>::infinity();
            for (const int point_index : candidates) {
                const double candidate_squared = static_cast<double>(
                    squared_distance(cloud->points[static_cast<std::size_t>(point_index)]));
                if (candidate_squared > near_squared && candidate_squared < best_squared) {
                    best_squared = candidate_squared;
                    best_index = point_index;
                }
            }
            if (best_index < 0) {
                continue;
            }

            const float distance = static_cast<float>(std::sqrt(best_squared));
            const pcl::PointXYZ& hit =
                cloud->points[static_cast<std::size_t>(best_index)];
            const Eigen::Vector3d hit_local = rotation.conjugate() *
                (Eigen::Vector3d(hit.x, hit.y, hit.z) - translation);
            const double longitude = std::atan2(-hit_local.y(), hit_local.x());
            const double latitude =
                std::atan2(hit_local.z(), std::hypot(hit_local.x(), hit_local.y()));
            const double projected_column =
                (longitude / (2.0 * pi) + 0.5) * width - 0.5;
            const double projected_row =
                (0.5 - latitude / pi) * height - 0.5;
            const int target_column = static_cast<int>(std::round(projected_column));
            const int target_row = static_cast<int>(std::round(projected_row));
            if (target_column < 0 || target_column >= width ||
                target_row < 0 || target_row >= height) {
                continue;
            }
            float& current = reprojected_depth(target_row, target_column);
            if (current >= 0.0F && current <= distance) {
                continue;
            }
            current = distance;
            ++projected;
        }
    }
    if (projected == 0U) {
        throw std::runtime_error("No surface points project into panorama depth map");
    }
    if (const char* dump_path = std::getenv("NAVVIS_RECON_SPARSE_RAW_DEPTH_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary | std::ios::trunc);
        dump.write(
            reinterpret_cast<const char*>(reprojected_depth.data),
            static_cast<std::streamsize>(
                reprojected_depth.total() * reprojected_depth.elemSize()));
        if (!dump) {
            throw std::runtime_error("Cannot write raw sparse-depth regression dump");
        }
    }

    cv::Mat1f measured = cv::Mat1f::zeros(height, width);
    cv::Mat1b valid = cv::Mat1b::zeros(height, width);
    for (int row = 0; row < height; ++row) {
        for (int column = 0; column < width; ++column) {
            const float depth = reprojected_depth(row, column);
            if (!(depth > 0.0F) || !std::isfinite(depth)) {
                continue;
            }
            // DepthMapPointCloudRaytracer multiplies in float before the
            // truncating integer conversion. Seven values in this regression
            // round up to an exact integer millimetre in that multiply.
            const float millimetres_float = depth * 1000.0F;
            const long truncated = static_cast<long>(millimetres_float);
            const std::uint16_t millimetres = static_cast<std::uint16_t>(
                std::clamp(truncated, 0L, 65535L));
            measured(row, column) = static_cast<float>(
                static_cast<double>(millimetres) * 0.001);
            valid(row, column) = millimetres != 0U ? 255U : 0U;
        }
    }
    if (const char* dump_path = std::getenv("NAVVIS_RECON_SPARSE_DEPTH_DUMP")) {
        std::ofstream dump(dump_path, std::ios::binary | std::ios::trunc);
        dump.write(
            reinterpret_cast<const char*>(measured.data),
            static_cast<std::streamsize>(measured.total() * measured.elemSize()));
        if (!dump) {
            throw std::runtime_error("Cannot write sparse-depth regression dump");
        }
    }
    return GaussNewtonDepthMapOptimizer::optimizeDouble(measured, valid, 1.0);
}

ImageStitcher::ImageStitcher(PanoramaOptions options) : options_(std::move(options)) {}

WarpedImage ImageStitcher::warp(
    const cv::Mat3b& image, const Camera& camera, const Pose& panorama_pose,
    int width, int height) {
    constexpr float pi = 3.14159265358979323846F;
    cv::Mat1f map_x(height, width);
    cv::Mat1f map_y(height, width);
    cv::Mat1f mask(height, width, 0.0F);
    for (int y = 0; y < height; ++y) {
        const float latitude =
            (0.5F - (static_cast<float>(y) + 0.5F) / static_cast<float>(height)) * pi;
        for (int x = 0; x < width; ++x) {
            const float longitude =
                ((static_cast<float>(x) + 0.5F) / static_cast<float>(width) - 0.5F) *
                2.0F * pi;
            const Vec3f ray_panorama(std::cos(latitude) * std::sin(longitude),
                                     std::sin(latitude),
                                     std::cos(latitude) * std::cos(longitude));
            const Vec3f ray_world = panorama_pose.rotation * ray_panorama;
            const Vec3f ray_camera = camera.world_from_camera.rotation.conjugate() * ray_world;
            if (ray_camera.z() <= 1.0e-8F) {
                map_x(y, x) = -1.0F;
                map_y(y, x) = -1.0F;
                continue;
            }
            map_x(y, x) = camera.fx * ray_camera.x() / ray_camera.z() + camera.cx;
            map_y(y, x) = camera.fy * ray_camera.y() / ray_camera.z() + camera.cy;
            if (camera.inside(map_x(y, x), map_y(y, x))) {
                mask(y, x) = 1.0F;
            }
        }
    }
    cv::Mat3b warped8;
    cv::remap(image, warped8, map_x, map_y, cv::INTER_LINEAR, cv::BORDER_CONSTANT);
    cv::Mat3f warped;
    warped8.convertTo(warped, CV_32FC3, 1.0 / 255.0);
    return {warped, mask};
}

cv::Mat3f ImageStitcher::stitch(
    const std::vector<cv::Mat3b>& images,
    const std::vector<Camera>& cameras,
    const Pose& panorama_pose) const {
    if (images.size() != cameras.size() || images.empty()) {
        throw std::invalid_argument("images and cameras must have equal non-zero size");
    }
    std::vector<cv::Mat3f> warped_images;
    std::vector<cv::Mat1f> masks;
    for (std::size_t i = 0; i < images.size(); ++i) {
        auto warped = warp(images[i], cameras[i], panorama_pose, options_.width, options_.height);
        warped_images.push_back(std::move(warped.image));
        masks.push_back(std::move(warped.mask));
    }
    std::vector<cv::Mat3b> exposure_images;
    std::vector<cv::Mat1b> exposure_masks;
    exposure_images.reserve(warped_images.size());
    exposure_masks.reserve(masks.size());
    for (std::size_t i = 0; i < warped_images.size(); ++i) {
        cv::Mat3b image;
        cv::Mat1b mask;
        warped_images[i].convertTo(image, CV_8UC3, 255.0);
        masks[i].convertTo(mask, CV_8U, 255.0);
        exposure_images.push_back(std::move(image));
        exposure_masks.push_back(std::move(mask));
    }
    const auto gains =
        ExposureCompensatorSoftConstraint::estimateGains(exposure_images, exposure_masks);
    for (std::size_t i = 0; i < warped_images.size(); ++i) {
        cv::multiply(warped_images[i], cv::Scalar(
            gains[i][0], gains[i][1], gains[i][2]), warped_images[i]);
    }
    const auto seam_masks = SeamMaskPreparer::prepare(masks, options_.seam_finder);
    cv::Mat3f panorama = MultiBandBlender::blend(warped_images, seam_masks, options_.multiband_levels);
    const cv::Mat1b valid = unionMask(masks);
    return options_.floor_filling ? FloorFiller::fill(panorama, valid)
                                  : PyramidInpainting::fillImage(panorama, valid);
}

PointCloudRenderer::PointCloudRenderer(SurfelRenderingOptions options)
    : options_(std::move(options)) {}

PointCloudRenderResult PointCloudRenderer::render(
    const std::vector<ColoredPoint>& cloud, const Pose& panorama_pose) const {
    constexpr float pi = 3.14159265358979323846F;
    cv::Mat3f accumulated = cv::Mat3f::zeros(options_.height, options_.width);
    cv::Mat1f weight = cv::Mat1f::zeros(options_.height, options_.width);
    cv::Mat1f depth(options_.height, options_.width, std::numeric_limits<float>::infinity());
    for (const auto& point : cloud) {
        if (!point.has_color) {
            continue;
        }
        const Vec3f local = panorama_pose.inverseApply(point.xyz);
        const float distance = local.norm();
        if (distance < options_.near_distance || distance > options_.far_distance) {
            continue;
        }
        const float longitude = std::atan2(local.x(), local.z());
        const float latitude = std::asin(std::clamp(local.y() / distance, -1.0F, 1.0F));
        const int center_x = static_cast<int>((longitude / (2.0F * pi) + 0.5F) * options_.width);
        const int center_y = static_cast<int>((0.5F - latitude / pi) * options_.height);
        const int radius = std::max(
            1, static_cast<int>(options_.surfel_radius / distance * options_.width / (2.0F * pi)));
        int minor_radius = radius;
        float ellipse_angle = 0.0F;
        if (point.has_normal) {
            const Vec3f normal_local = panorama_pose.rotation.conjugate() * point.normal;
            const Vec3f view_direction = -local.normalized();
            const float incidence = std::abs(normal_local.dot(view_direction));
            minor_radius = std::max(1, static_cast<int>(radius * std::max(incidence, 0.15F)));
            ellipse_angle = std::atan2(normal_local.y(), normal_local.x());
        }
        const float cosine = std::cos(ellipse_angle);
        const float sine = std::sin(ellipse_angle);
        for (int y = std::max(0, center_y - radius);
             y <= std::min(options_.height - 1, center_y + radius); ++y) {
            for (int x_unwrapped = center_x - radius; x_unwrapped <= center_x + radius; ++x_unwrapped) {
                const float dx = static_cast<float>(x_unwrapped - center_x);
                const float dy = static_cast<float>(y - center_y);
                const float major_coordinate = cosine * dx + sine * dy;
                const float minor_coordinate = -sine * dx + cosine * dy;
                if (major_coordinate * major_coordinate / static_cast<float>(radius * radius) +
                        minor_coordinate * minor_coordinate /
                            static_cast<float>(minor_radius * minor_radius) >
                    1.0F) {
                    continue;
                }
                const int x = (x_unwrapped % options_.width + options_.width) % options_.width;
                if (distance > depth(y, x) + options_.surfel_radius) {
                    continue;
                }
                const float sample_weight = 1.0F / (1.0F + distance * distance);
                accumulated(y, x) += cv::Vec3f(
                    sample_weight * static_cast<float>(point.rgb[0]) / 255.0F,
                    sample_weight * static_cast<float>(point.rgb[1]) / 255.0F,
                    sample_weight * static_cast<float>(point.rgb[2]) / 255.0F);
                weight(y, x) += sample_weight;
                depth(y, x) = std::min(depth(y, x), distance);
            }
        }
    }
    cv::Mat1b valid = weight > 0.0F;
    for (int y = 0; y < accumulated.rows; ++y) {
        for (int x = 0; x < accumulated.cols; ++x) {
            if (valid(y, x)) {
                accumulated(y, x) /= weight(y, x);
            }
        }
    }
    return {PyramidInpainting::fillImage(accumulated, valid), depth};
}

}  // namespace navvis_recon
