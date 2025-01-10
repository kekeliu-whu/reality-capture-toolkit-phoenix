#include <pcl/io/pcd_io.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <Eigen/Eigen>
#include <iostream>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/eigen.hpp>
#include <opencv2/opencv.hpp>

int main() {
  ////////////////////////////////////////////////////////////////////////
  ////                        Edit by manaul                           ///
  ////////////////////////////////////////////////////////////////////////
  // 相机内参矩阵K
  cv::Mat K = (cv::Mat_<double>(3, 3) << 1473.1956665797843, 0.0, 1785.8248915738964, 0.0, 1473.952880308936, 2327.6491626317593, 0.0, 0.0, 1.0);
  // 畸变系数D
  cv::Mat D = (cv::Mat_<double>(1, 4) << 0.01600925150571167, 0.007454388315425191, -0.003923151572616989, 0.0009809063388811954);

  // 图像点
  std::vector<cv::Point2f> imgPoints = {
      cv::Point2f(896, 1876),
      cv::Point2f(886, 2318),
      cv::Point2f(1463, 2305),
      cv::Point2f(1465, 1883)};

  // 三维点
  std::vector<cv::Point3f> objPoints = {
      cv::Point3f(-0.409000009298, 2.51699995995, 0.582000017166),
      cv::Point3f(-0.0790000036359, 2.56999993324, -0.0710000023246),
      cv::Point3f(0.785000026226, 2.38000011444, 0.361000001431),
      cv::Point3f(0.453999996185, 2.32699990273, 0.990000009537)};

  std::string ply_filename   = R"(D:\data\20\RawData\LidarImgData_right_4_sum_60__20241129_144711\4.ply)";
  std::string image_filename = R"(D:\data\20\RawData\LidarImgData_right_4_sum_60__20241129_144711\4.png)";
  //////////////////////////////////////////////////////////////////////////
  ////                        Edit by manaul                           ///
  //////////////////////////////////////////////////////////////////////////

  // 将畸变矫正为无畸变点
  std::vector<cv::Point2f> undistorted;
  cv::fisheye::undistortPoints(imgPoints, undistorted, K, D, {}, K);

  // 相机的初始位姿估计
  cv::Mat rvec, tvec;

// 使用solvePnP求解相机的位姿
#ifdef __linux__
  cv::solvePnPRansac(objPoints, undistorted, K, {}, rvec, tvec, {}, 100, 50);
#else
  cv::UsacParams usac_params;
  usac_params.threshold = 50;
  cv::solvePnPRansac(objPoints, undistorted, K, {}, rvec, tvec, {}, usac_params);
#endif

  // 计算重投影误差
  std::vector<cv::Point2f> projectedPoints;
  cv::fisheye::projectPoints(objPoints, projectedPoints, rvec, tvec, K, D);
  double pixel_error = 0;
  for (size_t i = 0; i < projectedPoints.size(); i++) {
    cv::Point2f diff = projectedPoints[i] - imgPoints[i];
    std::cout << "Point " << i << ": " << sqrt(diff.x * diff.x + diff.y * diff.y) << std::endl;
  }

  // 输出结果
  std::cout << "Rotation Vector:" << std::endl
            << rvec << std::endl;
  std::cout << "Translation Vector:" << std::endl
            << tvec << std::endl;

  cv::Mat rotationMatrix;
  cv::Rodrigues(rvec, rotationMatrix);

  // 将 OpenCV 的旋转矩阵转换为 Eigen 的旋转矩阵
  Eigen::Matrix3d eigenRotationMatrix;
  Eigen::Vector3d eigenTranslationVector;
  cv::cv2eigen(rotationMatrix, eigenRotationMatrix);
  cv::cv2eigen(tvec, eigenTranslationVector);

  std::cout << eigenRotationMatrix << std::endl;
  std::cout << eigenTranslationVector.transpose() << std::endl;

  pcl::PointCloud<pcl::PointXYZRGB>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZRGB>);
  int ok = pcl::io::loadPLYFile(ply_filename, *cloud);
  if (ok != 0) {
    std::cout << ok << " " << errno << std::endl;
    exit(1);
  }

  cv::Mat img = cv::imread(image_filename);
  for (auto& e : cloud->points) {
    e.b = e.g = e.r           = 0;
    Eigen::Vector3d pt_in_cam = eigenRotationMatrix * Eigen::Vector3d(e.x, e.y, e.z) + eigenTranslationVector;
    if (pt_in_cam.z() <= 0) {
      continue;
    }

    double a       = pt_in_cam.x() / pt_in_cam.z();
    double b       = pt_in_cam.y() / pt_in_cam.z();
    double r       = sqrt(a * a + b * b);
    double theta   = atan(r);
    double theta2  = theta * theta;
    double theta4  = theta2 * theta2;
    double theta6  = theta4 * theta2;
    double theta8  = theta6 * theta2;
    double theta_d = theta * (1 + D.at<double>(0) * theta2 + D.at<double>(1) * theta4 + D.at<double>(2) * theta6 + D.at<double>(3) * theta8);

    double x_ = (theta_d / r) * a;
    double y_ = (theta_d / r) * b;

    double u = K.at<double>(0, 0) * x_ + K.at<double>(0, 2);
    double v = K.at<double>(1, 1) * y_ + K.at<double>(1, 2);

    if (u >= 0 && u < img.cols && v >= 0 && v < img.rows) {
      cv::Vec3b color = img.at<cv::Vec3b>(v, u);
      e.b             = color[0];
      e.g             = color[1];
      e.r             = color[2];
    }
  }

  pcl::io::savePLYFileBinary(ply_filename + "-colored.ply", *cloud);

  // l2c -> c2l
  eigenTranslationVector = -(eigenRotationMatrix.transpose() * eigenTranslationVector);
  eigenRotationMatrix.transposeInPlace();

  std::cout << eigenRotationMatrix << std::endl;
  std::cout << Eigen::Quaterniond(eigenRotationMatrix).coeffs().transpose() << std::endl;
  std::cout << eigenTranslationVector.transpose() << std::endl;

  return 0;
}