#pragma once

#include <opencv2/core.hpp>

#include <cstddef>
#include <string>
#include <vector>

namespace navvis_recon {

struct BlurRegion {
    cv::Rect rectangle;
    int kernel_size = 31;
};

struct ImagePostprocessingOptions {
    enum class Preset { HighQuality, Fast, Plain } preset = Preset::HighQuality;
    enum class WhiteBalance { Camera, Automatic, Custom, PerPanorama } white_balance = WhiteBalance::Camera;
    enum class Denoise { AdaptiveWavelet, Off, ManualWavelet, Nlm, NlmFused } denoise = Denoise::NlmFused;
    float denoise_strength = 1.0F;
    float sharpen = 0.5F;
    float exposure_stops = 0.0F;
    cv::Vec3f custom_white_balance = cv::Vec3f(1.0F, 1.0F, 1.0F);  // RGB
    std::vector<float> hdr_ev_shifts = {0.0F, 1.5F, 3.0F};
    std::size_t maximum_jpeg_bytes = 0U;
};

class ImagePostprocessor {
public:
    explicit ImagePostprocessor(ImagePostprocessingOptions options);

    [[nodiscard]] cv::Mat3f processLinearRgb(
        const cv::Mat3f& linear_rgb,
        const cv::Vec3f& camera_white_balance = cv::Vec3f(1.0F, 1.0F, 1.0F),
        const std::vector<float>& vignetting_polynomial = {1.0F},
        const std::vector<BlurRegion>& blur_regions = {}) const;

    [[nodiscard]] std::vector<std::uint8_t> encodeJpeg(const cv::Mat3f& image_bgr) const;

    static cv::Mat3f applyWhiteBalance(
        const cv::Mat3f& linear_rgb, ImagePostprocessingOptions::WhiteBalance mode,
        const cv::Vec3f& camera_gains, const cv::Vec3f& custom_gains);
    static cv::Mat3f applyVignetting(
        const cv::Mat3f& image, const std::vector<float>& polynomial);
    static cv::Mat3f exposureFusion(
        const cv::Mat3f& linear_rgb, const std::vector<float>& ev_shifts);
    static cv::Mat3f denoiseImage(
        const cv::Mat3f& image_bgr, ImagePostprocessingOptions::Denoise mode, float strength);
    static cv::Mat3f unsharpMask(const cv::Mat3f& image, float amount);
    static cv::Mat3f blurRegions(const cv::Mat3f& image, const std::vector<BlurRegion>& regions);

private:
    ImagePostprocessingOptions options_;
};

}  // namespace navvis_recon
