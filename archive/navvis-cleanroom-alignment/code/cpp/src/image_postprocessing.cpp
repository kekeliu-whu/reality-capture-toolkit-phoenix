#include "navvis_recon/image_postprocessing.hpp"

#include <opencv2/imgcodecs.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/photo.hpp>

#include <algorithm>
#include <cmath>

namespace navvis_recon {
namespace {

cv::Mat3f clamp01(const cv::Mat3f& input) {
    cv::Mat3f output;
    cv::max(input, 0.0, output);
    cv::min(output, 1.0, output);
    return output;
}

cv::Mat3b to8Bit(const cv::Mat3f& input) {
    cv::Mat3b output;
    clamp01(input).convertTo(output, CV_8UC3, 255.0);
    return output;
}

cv::Mat3f srgbFromLinear(const cv::Mat3f& linear) {
    cv::Mat3f output(linear.size());
    for (int y = 0; y < linear.rows; ++y) {
        for (int x = 0; x < linear.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = std::clamp(linear(y, x)[channel], 0.0F, 1.0F);
                output(y, x)[channel] = value <= 0.0031308F
                                              ? 12.92F * value
                                              : 1.055F * std::pow(value, 1.0F / 2.4F) - 0.055F;
            }
        }
    }
    return output;
}

}  // namespace

ImagePostprocessor::ImagePostprocessor(ImagePostprocessingOptions options)
    : options_(std::move(options)) {}

cv::Mat3f ImagePostprocessor::applyWhiteBalance(
    const cv::Mat3f& linear_rgb, ImagePostprocessingOptions::WhiteBalance mode,
    const cv::Vec3f& camera_gains, const cv::Vec3f& custom_gains) {
    cv::Vec3f gains = camera_gains;
    if (mode == ImagePostprocessingOptions::WhiteBalance::Custom) {
        gains = custom_gains;
    } else if (mode == ImagePostprocessingOptions::WhiteBalance::Automatic) {
        cv::Scalar means = cv::mean(linear_rgb);
        const float average = static_cast<float>((means[0] + means[1] + means[2]) / 3.0);
        for (int channel = 0; channel < 3; ++channel) {
            gains[channel] = average / std::max(static_cast<float>(means[channel]), 1.0e-6F);
        }
    }
    cv::Mat3f output = linear_rgb.clone();
    for (int y = 0; y < output.rows; ++y) {
        for (int x = 0; x < output.cols; ++x) {
            output(y, x) = output(y, x).mul(gains);
        }
    }
    return clamp01(output);
}

cv::Mat3f ImagePostprocessor::applyVignetting(
    const cv::Mat3f& image, const std::vector<float>& polynomial) {
    cv::Mat3f output = image.clone();
    const float center_x = 0.5F * static_cast<float>(image.cols - 1);
    const float center_y = 0.5F * static_cast<float>(image.rows - 1);
    for (int y = 0; y < image.rows; ++y) {
        for (int x = 0; x < image.cols; ++x) {
            const float nx = (static_cast<float>(x) - center_x) / std::max(center_x, 1.0F);
            const float ny = (static_cast<float>(y) - center_y) / std::max(center_y, 1.0F);
            const float radius_squared = nx * nx + ny * ny;
            float attenuation = 0.0F;
            float power = 1.0F;
            for (const float coefficient : polynomial) {
                attenuation += coefficient * power;
                power *= radius_squared;
            }
            output(y, x) /= std::max(attenuation, 0.1F);
        }
    }
    return clamp01(output);
}

cv::Mat3f ImagePostprocessor::exposureFusion(
    const cv::Mat3f& linear_rgb, const std::vector<float>& ev_shifts) {
    std::vector<cv::Mat> exposures;
    for (const float shift : ev_shifts) {
        cv::Mat3f shifted = clamp01(linear_rgb * std::pow(2.0F, shift));
        cv::Mat3f srgb = srgbFromLinear(shifted);
        cv::Mat3f bgr;
        cv::cvtColor(srgb, bgr, cv::COLOR_RGB2BGR);
        exposures.push_back(to8Bit(bgr));
    }
    cv::Mat fused;
    cv::createMergeMertens()->process(exposures, fused);
    return clamp01(fused);
}

cv::Mat3f ImagePostprocessor::denoiseImage(
    const cv::Mat3f& image_bgr, ImagePostprocessingOptions::Denoise mode, float strength) {
    if (mode == ImagePostprocessingOptions::Denoise::Off) {
        return image_bgr;
    }
    const cv::Mat3b input8 = to8Bit(image_bgr);
    if (mode == ImagePostprocessingOptions::Denoise::Nlm ||
        mode == ImagePostprocessingOptions::Denoise::NlmFused) {
        cv::Mat3b output8;
        // Reference high-quality path: h=4.6785717, template 7, search 17.
        // Both named NLM modes enter the same OpenCV stage in the binary.
        constexpr float h = 4.6785717F;
        cv::fastNlMeansDenoising(input8, output8, h, 7, 17);
        cv::Mat3f output;
        output8.convertTo(output, CV_32FC3, 1.0 / 255.0);
        return output;
    }
    // The binary names adaptive/manual wavelet modes but strips the private
    // wavelet implementation. This is the equivalent coefficient-shrink path.
    cv::Mat3b base8;
    cv::bilateralFilter(input8, base8, 7, 25.0, 5.0);
    cv::Mat3f base;
    base8.convertTo(base, CV_32FC3, 1.0 / 255.0);
    cv::Mat3f detail = image_bgr - base;
    const float threshold = 0.008F * std::max(strength, 0.1F);
    for (int y = 0; y < detail.rows; ++y) {
        for (int x = 0; x < detail.cols; ++x) {
            for (int channel = 0; channel < 3; ++channel) {
                const float value = detail(y, x)[channel];
                detail(y, x)[channel] = std::copysign(std::max(std::abs(value) - threshold, 0.0F), value);
            }
        }
    }
    return clamp01(base + detail);
}

cv::Mat3f ImagePostprocessor::unsharpMask(const cv::Mat3f& image, float amount) {
    if (amount <= 0.0F) {
        return image;
    }
    cv::Mat3f blurred;
    // The binary forms 1.5 * source - 0.5 * Gaussian(source, sigma=3)
    // for its default amount of 0.5.
    cv::GaussianBlur(image, blurred, cv::Size(), 3.0);
    return clamp01((1.0F + amount) * image - amount * blurred);
}

cv::Mat3f ImagePostprocessor::blurRegions(
    const cv::Mat3f& image, const std::vector<BlurRegion>& regions) {
    cv::Mat3f result = image.clone();
    const cv::Rect bounds(0, 0, image.cols, image.rows);
    for (const auto& region : regions) {
        const cv::Rect clipped = region.rectangle & bounds;
        if (clipped.empty()) {
            continue;
        }
        int kernel = std::max(3, region.kernel_size);
        kernel |= 1;
        cv::GaussianBlur(result(clipped), result(clipped), cv::Size(kernel, kernel), 0.0);
    }
    return result;
}

cv::Mat3f ImagePostprocessor::processLinearRgb(
    const cv::Mat3f& linear_rgb, const cv::Vec3f& camera_white_balance,
    const std::vector<float>& vignetting_polynomial,
    const std::vector<BlurRegion>& blur_regions) const {
    cv::Mat3f image = applyWhiteBalance(
        linear_rgb, options_.white_balance, camera_white_balance, options_.custom_white_balance);
    image = applyVignetting(image, vignetting_polynomial);
    image = clamp01(image * std::pow(2.0F, options_.exposure_stops));
    if (options_.preset == ImagePostprocessingOptions::Preset::Plain) {
        cv::cvtColor(image, image, cv::COLOR_RGB2BGR);
    } else {
        const std::vector<float> shifts =
            options_.preset == ImagePostprocessingOptions::Preset::Fast
                ? std::vector<float>{0.0F}
                : options_.hdr_ev_shifts;
        image = exposureFusion(image, shifts);
        image = denoiseImage(image, options_.denoise, options_.denoise_strength);
        image = unsharpMask(image, options_.sharpen);
    }
    return blurRegions(image, blur_regions);
}

std::vector<std::uint8_t> ImagePostprocessor::encodeJpeg(const cv::Mat3f& image_bgr) const {
    const cv::Mat3b image8 = to8Bit(image_bgr);
    std::vector<std::uint8_t> encoded;
    for (int quality = 95; quality >= 35; quality -= 5) {
        if (!cv::imencode(".jpg", image8, encoded, {cv::IMWRITE_JPEG_QUALITY, quality})) {
            throw std::runtime_error("JPEG encoding failed");
        }
        if (options_.maximum_jpeg_bytes == 0U || encoded.size() <= options_.maximum_jpeg_bytes) {
            return encoded;
        }
    }
    return encoded;
}

}  // namespace navvis_recon
