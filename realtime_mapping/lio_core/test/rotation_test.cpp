//
// Created by youyuan on 23-12-5.
//

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <Eigen/Core>
#include <Eigen/Dense>
#include <chrono>
#include <cmath>
#include <iostream>

TEST(RotationTest, Quantanion)
{
  // 定义旋转角度（以弧度为单位）
  double yaw = 90.0 * M_PI / 180.0;
  // 创建旋转矩阵
  Eigen::Matrix3f rotation_matrix;
  // 填充旋转矩阵
  rotation_matrix << std::cos(yaw), -std::sin(yaw), 0, std::sin(yaw), std::cos(yaw), 0, 0, 0, 1;

  Eigen::Quaternionf q(rotation_matrix);

  std::cout.precision(20);
  std::cout << "q:" << q.coeffs() << std::endl;
  q.normalize();
  std::cout << "q:" << q.coeffs() << std::endl;

  Eigen::Vector3f t = Eigen::Vector3f::Zero();
  t(0) = 1;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 1; i <= (int)1e8; ++i)
  {
    t = q * t;
    //
    //    if (i % (int)1e7 == 0)
    //    {
    //      std::cout << "t:" << t.transpose() << std::endl;
    //    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "t:" << t.transpose() << std::endl;
  std::cout << "耗时：" << duration << " 毫秒" << std::endl;
}

TEST(RotationTest, RotationMatrix)
{
  // 定义旋转角度（以弧度为单位）
  double yaw = 90.0 * M_PI / 180.0;
  // 创建旋转矩阵
  Eigen::Matrix3d rotation_matrix;
  // 填充旋转矩阵
  rotation_matrix << std::cos(yaw), -std::sin(yaw), 0, std::sin(yaw), std::cos(yaw), 0, 0, 0, 1;
  Eigen::Quaterniond q(rotation_matrix);
  Eigen::Matrix3d rot = q.toRotationMatrix();

  Eigen::Vector3d t = Eigen::Vector3d::Zero();
  t(0) = 1;

  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 1; i <= (int)1e8; ++i)
  {
    t = rot * t;
    //
    //    if (i % (int)1e7 == 0)
    //    {
    //      std::cout << "t:" << t.transpose() << std::endl;
    //    }
  }
  auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
  std::cout << "t:" << t.transpose() << std::endl;
  std::cout << "耗时：" << duration << " 毫秒" << std::endl;
}