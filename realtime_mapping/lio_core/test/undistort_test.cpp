
#include "interface/undistort.h"
#include <gtest/gtest.h>
#include <pcl/filters/uniform_sampling.h>
#include <pcl/filters/voxel_grid.h>
#include <pcl/io/ply_io.h>
#include <pcl/point_types.h>
#include <pcl/visualization/pcl_visualizer.h>
#include <pcl/visualization/point_cloud_geometry_handlers.h>
#include <Eigen/Dense>
#include <chrono>
#include <iostream>
#include <limits>
#include <pcl/visualization/impl/point_cloud_geometry_handlers.hpp>
#include <string>
#include "common/common_struct.h"
#include "common/math_utils.h"
Eigen::MatrixXd readCSV(const std::string &filename)
{
  std::vector<std::vector<double>> data;

  ifstream file(filename);
  if (!file.is_open())
  {
    cerr << "Error opening file: " << filename << endl;
    return Eigen::MatrixXd();
  }

  std::string line;
  while (getline(file, line))
  {
    std::stringstream ss(line);
    std::vector<double> row;
    std::string cell;

    while (getline(ss, cell, ','))
    {
      row.push_back(stod(cell));  // 将字符串转换为双精度浮点数
    }
    data.push_back(row);
  }

  file.close();

  int rows = data.size();
  int cols = data[0].size();
  Eigen::MatrixXd matrix(rows, cols);
  for (int i = 0; i < rows; ++i)
  {
    for (int j = 0; j < cols; ++j)
    {
      matrix(i, j) = (data[i][j]);
    }
  }
  return matrix;
}
void writeCSV(const Eigen::MatrixXd &matrix)
{
  std::ofstream file("matrix.csv");

  if (file.is_open())
  {
    for (int i = 0; i < matrix.rows(); ++i)
    {
      for (int j = 0; j < matrix.cols(); ++j)
      {
        if (j > 0)
        {
          file << ",";
        }
        file << matrix(i, j);
      }
      file << "\n";
    }

    file.close();
    std::cout << "Matrix has been written to matrix.csv" << std::endl;
  }
  else
  {
    std::cerr << "Unable to open file for writing." << std::endl;
  }

  return;
}

lixel::PointCloud getLidarData(const std::string &file_path)
{
  lixel::PointCloud points;
  auto matrix = readCSV(file_path);
  for (int i = 0; i < matrix.rows(); ++i)
  {
    lixel::PointT point;
    point.timestamp = matrix(i, 0);
    point.x = matrix(i, 1);
    point.y = matrix(i, 2);
    point.z = matrix(i, 3);
    points.push_back(point);
    // std::cout <<std::fixed<<  point.timestamp << ' '<< point.x << ' ' << point.y << ' ' <<
    // point.z << std::endl;
  }
  return points;
}

lixel::StatePredict getPosAtt(const std::string &file_path)
{
  lixel::StatePredict state_predict;
  auto matrix = readCSV(file_path);

  lixel::PosAtt pos_att;
  lixel::V3F axis_angle;
  for (int i = 0; i < matrix.rows(); ++i)
  {
    pos_att.timestamp = matrix(i, 0);
    pos_att.pos << matrix(i, 1), matrix(i, 2), matrix(i, 3);
    axis_angle << matrix(i, 7), matrix(i, 8), matrix(i, 9);
    Eigen::AngleAxisf angle_axis(axis_angle.norm(), axis_angle.normalized());
    lixel::QUATF quat(angle_axis);
    pos_att.quat = quat;
    state_predict.push_back(pos_att);
  }
  return state_predict;
}

int main(int argc, char **argv)
{
  auto states_predict = getPosAtt("../data/state_predict.csv");

  auto raw_points = getLidarData("../data/raw_lidar.csv");
  auto undistort_points = getLidarData("../data/undistort_lidar.csv");
  lixel::PointCloud output;
  auto start = std::chrono::high_resolution_clock::now();
  for (int i = 0; i < 1000; i++)
  {
    auto error = undistort(raw_points, states_predict, output);
  }
  auto end = std::chrono::high_resolution_clock::now();
  std::chrono::duration<double> duration = end - start;
  std::cout << "Function execution time: " << duration.count() << " seconds" << std::endl;
  Eigen::MatrixXd differenceMatrix(undistort_points.size(), 3);
  for (int i = 0; i < output.size(); ++i)
  {
    auto error = output[i].getArray3fMap() - undistort_points[i].getArray3fMap();
    differenceMatrix.row(i) << error.x(), error.y(), error.z();
  }
  writeCSV(differenceMatrix);
  return 0;
}