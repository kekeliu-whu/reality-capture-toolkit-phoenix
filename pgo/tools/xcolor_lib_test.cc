#include "xcolor_lib.h"

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

}  // namespace
}  // namespace xcolor
