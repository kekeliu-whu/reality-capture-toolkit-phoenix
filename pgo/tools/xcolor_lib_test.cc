#include "xcolor_lib.h"
#include "xsfm_post_depth.h"

#include <algorithm>
#include <array>
#include <limits>

#include <gtest/gtest.h>

namespace xcolor {
namespace {

TEST(XColorMaskTest, AllowsNonZeroPixelsAndRejectsZeroPixels) {
  cv::Mat mask = (cv::Mat_<uint8_t>(2, 2) << 0, 255, 1, 128);

  EXPECT_FALSE(IsMaskPixelAllowed(mask, Eigen::Vector2d(0.0, 0.0)));
  EXPECT_TRUE(IsMaskPixelAllowed(mask, Eigen::Vector2d(1.0, 0.0)));
  EXPECT_TRUE(IsMaskPixelAllowed(mask, Eigen::Vector2d(0.0, 1.0)));
  EXPECT_TRUE(IsMaskPixelAllowed(mask, Eigen::Vector2d(1.0, 1.0)));
}

TEST(XColorMaskTest, AllowsPixelsWhenMaskIsAbsent) {
  EXPECT_TRUE(IsMaskPixelAllowed(cv::Mat(), Eigen::Vector2d(100.0, 100.0)));
}

TEST(XColorMaskTest, RejectsPixelsOutsideMask) {
  const cv::Mat mask(2, 2, CV_8UC1, cv::Scalar(255));

  EXPECT_FALSE(IsMaskPixelAllowed(mask, Eigen::Vector2d(-1.0, 0.0)));
  EXPECT_FALSE(IsMaskPixelAllowed(mask, Eigen::Vector2d(2.0, 0.0)));
  EXPECT_FALSE(IsMaskPixelAllowed(mask, Eigen::Vector2d(0.0, 2.0)));
}

TEST(XColorNormalTest, AcceptsCameraOnNormalSide) {
  EXPECT_TRUE(IsNormalFacingCamera(Eigen::Vector3d(0.0, 0.0, 2.0),
                                   Eigen::Vector3d(0.0, 0.0, -1.0)));
}

TEST(XColorNormalTest, RejectsCameraBehindSurfaceAndAtTangent) {
  EXPECT_FALSE(IsNormalFacingCamera(Eigen::Vector3d(0.0, 0.0, 2.0),
                                    Eigen::Vector3d(0.0, 0.0, 1.0)));
  EXPECT_FALSE(IsNormalFacingCamera(Eigen::Vector3d(0.0, 0.0, 2.0),
                                    Eigen::Vector3d(1.0, 0.0, 0.0)));
}

TEST(XColorNormalTest, SkipsFilteringForMissingNormal) {
  EXPECT_TRUE(IsNormalFacingCamera(Eigen::Vector3d(0.0, 0.0, 2.0),
                                   Eigen::Vector3d::Zero()));
  EXPECT_TRUE(IsNormalFacingCamera(
      Eigen::Vector3d(0.0, 0.0, 2.0),
      Eigen::Vector3d::Constant(std::numeric_limits<double>::quiet_NaN())));
}

TEST(XColorDistanceTest, UsesDirectCameraDistance) {
  EXPECT_DOUBLE_EQ(ComputeCandidateDistance(Eigen::Vector3d(3.0, 4.0, 12.0)),
                   13.0);
}

TEST(XColorFisheyeDepthTest, RendersOpenCvFisheyeDepthAtExpectedPixels) {
  if (!xsfm_post::HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const std::vector<xsfm_post::DepthWorldPoint> points = {
      {0.0f, 0.0f, 2.0f}, {1.0f, 0.0f, 1.0f}};
  xsfm_post::CudaDepthRenderer renderer(points);
  const std::array<double, 9> identity = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  xsfm_post::OpenCVFisheyeIntrinsics camera;
  camera.width  = 120;
  camera.height = 100;
  camera.fx     = 50.0;
  camera.fy     = 50.0;
  camera.cx     = 50.0;
  camera.cy     = 50.0;

  const auto depth = renderer.RenderFisheye(
      identity, zero, camera, 0.01f, 1024, 30.0f, false);
  ASSERT_EQ(depth.width, 120);
  ASSERT_EQ(depth.height, 100);
  EXPECT_FLOAT_EQ(depth.depth[50 * depth.width + 50], 2.0f);
  // atan2(1, 1) = pi/4, so u = 50 + 50*pi/4 and the renderer rounds u-0.5.
  EXPECT_FLOAT_EQ(depth.depth[50 * depth.width + 89], 1.0f);
}

TEST(XColorFisheyeDepthTest, ReturnsDepthVisiblePointsInSourcePixels) {
  if (!xsfm_post::HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const std::vector<xsfm_post::DepthWorldPoint> points = {
      {0.0f, 0.0f, 2.0f},
      {0.01f, 0.0f, 3.0f},  // Occluded by point 0.
      {1.0f, 0.0f, 1.0f}};
  xsfm_post::CudaDepthRenderer renderer(points);
  const std::array<double, 9> identity = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  xsfm_post::OpenCVFisheyeIntrinsics camera;
  camera.width  = 120;
  camera.height = 100;
  camera.fx     = 50.0;
  camera.fy     = 50.0;
  camera.cx     = 50.0;
  camera.cy     = 50.0;

  const auto result = renderer.RenderFisheyeVisibility(identity,
                                                       zero,
                                                       camera,
                                                       240,
                                                       200,
                                                       0.01f,
                                                       1024,
                                                       30.0f,
                                                       0.15f,
                                                       false);
  EXPECT_TRUE(result.depth.empty());
  EXPECT_GT(result.contributing_count, 0);
  ASSERT_EQ(result.visible_points.size(), 2);
  const auto find_point = [&result](uint32_t index) {
    return std::find_if(result.visible_points.begin(),
                        result.visible_points.end(),
                        [index](const auto& point) {
                          return point.point_index == index;
                        });
  };
  const auto center = find_point(0);
  ASSERT_NE(center, result.visible_points.end());
  EXPECT_EQ(center->pixel_x, 100);
  EXPECT_EQ(center->pixel_y, 100);
  EXPECT_FLOAT_EQ(center->distance, 2.0f);
  EXPECT_EQ(find_point(1), result.visible_points.end());
  const auto off_axis = find_point(2);
  ASSERT_NE(off_axis, result.visible_points.end());
  EXPECT_EQ(off_axis->pixel_x, 178);
  EXPECT_EQ(off_axis->pixel_y, 100);
}

TEST(XColorFisheyeDepthTest, FiltersNormalsAndMaskOnGpu) {
  if (!xsfm_post::HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const std::vector<xsfm_post::DepthWorldPoint> points = {
      {0.0f, 0.0f, 2.0f}, {1.0f, 0.0f, 1.0f}};
  const std::vector<xsfm_post::DepthWorldNormal> normals = {
      {0.0f, 0.0f, -1.0f}, {0.0f, 0.0f, 1.0f}};
  xsfm_post::CudaDepthRenderer renderer(points, normals);
  const std::array<double, 9> identity = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  xsfm_post::OpenCVFisheyeIntrinsics camera;
  camera.width  = 120;
  camera.height = 100;
  camera.fx     = 50.0;
  camera.fy     = 50.0;
  camera.cx     = 50.0;
  camera.cy     = 50.0;

  auto visible = renderer.RenderFisheyeVisibility(
      identity, zero, camera, 240, 200, 0.01f, 1024, 30.0f, 0.15f, false);
  ASSERT_EQ(visible.visible_points.size(), 1);
  EXPECT_EQ(visible.visible_points.front().point_index, 0);

  std::vector<uint8_t> mask(240 * 200, 255);
  mask[100 * 240 + 100] = 0;
  renderer.SetVisibilityMask(mask.data(), 240, 200, 240);
  visible = renderer.RenderFisheyeVisibility(
      identity, zero, camera, 240, 200, 0.01f, 1024, 30.0f, 0.15f, false);
  EXPECT_TRUE(visible.visible_points.empty());
  renderer.ClearVisibilityMask();
}

TEST(XColorFisheyeDepthTest,
     SelectsBestGpuColorAndCompensatesValidExposure) {
  if (!xsfm_post::HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const std::vector<xsfm_post::DepthWorldPoint> points = {
      {0.0f, 0.0f, 2.0f}};
  const std::vector<xsfm_post::DepthWorldNormal> normals = {
      {0.0f, 0.0f, -1.0f}};
  xsfm_post::CudaDepthRenderer renderer(points, normals);
  renderer.ResetFusedColors();
  const std::array<double, 9> identity = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  xsfm_post::OpenCVFisheyeIntrinsics camera;
  camera.width  = 120;
  camera.height = 100;
  camera.fx     = 50.0;
  camera.fy     = 50.0;
  camera.cx     = 50.0;
  camera.cy     = 50.0;

  constexpr int source_width  = 240;
  constexpr int source_height = 200;
  std::vector<uint8_t> image(source_width * source_height * 3);
  for (size_t i = 0; i < image.size(); i += 3) {
    image[i]     = 10;
    image[i + 1] = 20;
    image[i + 2] = 30;
  }
  auto result = renderer.RenderFisheyeColor(identity,
                                            zero,
                                            camera,
                                            image.data(),
                                            source_width,
                                            source_height,
                                            source_width * 3,
                                            0.01f,
                                            1024,
                                            30.0f,
                                            0.15f,
                                            10,
                                            0,
                                            false);
  EXPECT_TRUE(result.depth.empty());

  std::fill(image.begin(), image.end(), 255);
  renderer.RenderFisheyeColor(identity,
                              zero,
                              camera,
                              image.data(),
                              source_width,
                              source_height,
                              source_width * 3,
                              0.01f,
                              1024,
                              30.0f,
                              0.15f,
                              10,
                              1,
                              false);
  for (size_t i = 0; i < image.size(); i += 3) {
    image[i]     = 15;
    image[i + 1] = 30;
    image[i + 2] = 45;
  }
  renderer.RenderFisheyeColor(identity,
                              zero,
                              camera,
                              image.data(),
                              source_width,
                              source_height,
                              source_width * 3,
                              0.01f,
                              1024,
                              30.0f,
                              0.15f,
                              10,
                              2,
                              false);
  const auto colors = renderer.DownloadFusedColors();
  ASSERT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0].blue, 12, 1);
  EXPECT_NEAR(colors[0].green, 25, 1);
  EXPECT_NEAR(colors[0].red, 37, 1);
  EXPECT_EQ(colors[0].view_count, 3);
}

TEST(XColorFisheyeDepthTest,
     UsesMultiViewConsensusAndRejectsAmbiguousOutliers) {
  if (!xsfm_post::HasCudaDevice()) {
    GTEST_SKIP() << "CUDA device is unavailable";
  }
  const std::vector<xsfm_post::DepthWorldPoint> points = {
      {0.0f, 0.0f, 2.0f}};
  const std::vector<xsfm_post::DepthWorldNormal> normals = {
      {0.0f, 0.0f, -1.0f}};
  xsfm_post::CudaDepthRenderer renderer(points, normals);
  const std::array<double, 9> identity = {
      1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0};
  const std::array<double, 3> zero = {0.0, 0.0, 0.0};
  xsfm_post::OpenCVFisheyeIntrinsics camera;
  camera.width  = 120;
  camera.height = 100;
  camera.fx     = 50.0;
  camera.fy     = 50.0;
  camera.cx     = 50.0;
  camera.cy     = 50.0;

  constexpr int source_width  = 240;
  constexpr int source_height = 200;
  std::vector<uint8_t> image(source_width * source_height * 3);
  const auto add_observation = [&](uint8_t blue,
                                   uint8_t green,
                                   uint8_t red,
                                   int view_index) {
    for (size_t i = 0; i < image.size(); i += 3) {
      image[i]     = blue;
      image[i + 1] = green;
      image[i + 2] = red;
    }
    renderer.RenderFisheyeColor(identity,
                                zero,
                                camera,
                                image.data(),
                                source_width,
                                source_height,
                                source_width * 3,
                                0.01f,
                                1024,
                                30.0f,
                                0.15f,
                                10,
                                view_index,
                                false);
  };
  const auto render_current_image = [&](int view_index) {
    renderer.RenderFisheyeColor(identity,
                                zero,
                                camera,
                                image.data(),
                                source_width,
                                source_height,
                                source_width * 3,
                                0.01f,
                                1024,
                                30.0f,
                                0.15f,
                                10,
                                view_index,
                                false);
  };

  // The green observation has almost the same luminance as the background,
  // so it can only be rejected by chromatic multi-view agreement.
  renderer.ResetFusedColors();
  add_observation(40, 80, 120, 0);
  add_observation(40, 80, 120, 1);
  add_observation(40, 80, 120, 2);
  add_observation(0, 149, 0, 3);
  auto colors = renderer.DownloadFusedColors();
  ASSERT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0].blue, 40, 1);
  EXPECT_NEAR(colors[0].green, 80, 1);
  EXPECT_NEAR(colors[0].red, 120, 1);
  EXPECT_EQ(colors[0].view_count, 4);

  // The sharpest single view can be close enough for the exposure-tolerant
  // consensus cluster while still having visibly wrong orange/red chroma.
  // Keep the dominant neutral observation instead of restoring that view.
  renderer.ResetFusedColors();
  add_observation(97, 122, 175, 8);
  add_observation(125, 135, 139, 9);
  add_observation(125, 135, 139, 10);
  add_observation(125, 135, 139, 11);
  colors = renderer.DownloadFusedColors();
  ASSERT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0].blue, 125, 1);
  EXPECT_NEAR(colors[0].green, 135, 1);
  EXPECT_NEAR(colors[0].red, 139, 1);
  EXPECT_EQ(colors[0].view_count, 4);

  // When geometrically equivalent views agree photometrically, transfer the
  // exact pixel from the view with stronger local high-frequency detail.
  renderer.ResetFusedColors();
  add_observation(100, 100, 100, 12);
  for (size_t i = 0; i < image.size(); i += 3) {
    image[i]     = 90;
    image[i + 1] = 100;
    image[i + 2] = 104;
  }
  constexpr int projected_x = 100;
  constexpr int projected_y = 100;
  const auto set_gray       = [&](int x, int y, uint8_t value) {
    const size_t offset =
        (static_cast<size_t>(y) * source_width + x) * 3;
    image[offset]     = value;
    image[offset + 1] = value;
    image[offset + 2] = value;
  };
  set_gray(projected_x - 1, projected_y, 20);
  set_gray(projected_x + 1, projected_y, 220);
  set_gray(projected_x, projected_y - 1, 20);
  set_gray(projected_x, projected_y + 1, 220);
  render_current_image(13);
  add_observation(100, 100, 100, 14);
  colors = renderer.DownloadFusedColors();
  ASSERT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0].blue, 90, 1);
  EXPECT_NEAR(colors[0].green, 100, 1);
  EXPECT_NEAR(colors[0].red, 104, 1);
  EXPECT_EQ(colors[0].view_count, 3);

  // A 2:2 split has no statistical majority and is omitted rather than
  // committing an arbitrary foreground/background projection.
  renderer.ResetFusedColors();
  add_observation(40, 80, 120, 4);
  add_observation(40, 80, 120, 5);
  add_observation(0, 149, 0, 6);
  add_observation(0, 149, 0, 7);
  colors = renderer.DownloadFusedColors();
  ASSERT_EQ(colors.size(), 1);
  EXPECT_EQ(colors[0].view_count, 0);

  // Smooth fusion returns the robust mean of a mutually consistent cluster
  // rather than switching between individual source pixels.
  renderer.ResetFusedColors();
  add_observation(40, 80, 120, 20);
  add_observation(42, 82, 122, 21);
  add_observation(44, 84, 124, 22);
  colors = renderer.DownloadFusedColors(true);
  ASSERT_EQ(colors.size(), 1);
  EXPECT_NEAR(colors[0].blue, 42, 2);
  EXPECT_NEAR(colors[0].green, 82, 2);
  EXPECT_NEAR(colors[0].red, 122, 2);
  EXPECT_EQ(colors[0].view_count, 3);
}

}  // namespace
}  // namespace xcolor
