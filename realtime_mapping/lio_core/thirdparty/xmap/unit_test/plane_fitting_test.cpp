#include <gtest/gtest.h>
#include <Eigen/Eigen>
#include <iostream>
#include <random>
#include <vector>

#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>

typedef Eigen::Vector3d V3D;
typedef pcl::PointXYZ PointT;
typedef pcl::PointCloud<PointT> PointCloud;

class TicToc {
 public:
  TicToc() { Tic(); }

  void Tic() { start_ = std::chrono::system_clock::now(); }

  double Toc() {
    end_ = std::chrono::system_clock::now();
    elapsed_seconds_ = end_ - start_;
    return elapsed_seconds_.count() * 1000;
  }

  double GetLastStop() { return elapsed_seconds_.count() * 1000; }

 private:
  std::chrono::time_point<std::chrono::system_clock> start_, end_;
  std::chrono::duration<double> elapsed_seconds_;
};

// 使用PCA进行平面拟合
V3D fitPlaneWithPCA(const std::vector<V3D>& points) {
  int numPoints = points.size();

  // 构建数据矩阵
  Eigen::MatrixXd dataMatrix(numPoints, 3);
  for (int i = 0; i < numPoints; i++) {
    dataMatrix.row(i) = points[i].transpose();
  }

  // 去除数据均值
  V3D mean = dataMatrix.colwise().mean();
  dataMatrix.rowwise() -= mean.transpose();

  // 计算协方差矩阵
  Eigen::MatrixXd covarianceMatrix = (dataMatrix.transpose() * dataMatrix) / (numPoints - 1);

  // 特征值分解
  Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd> eigensolver(covarianceMatrix);

  std::cout << "eigen value PCL:" << eigensolver.eigenvalues().transpose() << std::endl;
  // 获取最小特征值对应的特征向量作为平面法向量
  V3D normal = eigensolver.eigenvectors().col(0);
  normal.normalize();
  return normal;
}

// 使用最小二乘法进行平面拟合
V3D fitPlaneWithLeastSquares(const std::vector<V3D>& points) {
  int numPoints = points.size();

  // 构建系数矩阵A和观测值向量b
  Eigen::MatrixXd A(numPoints, 3);
  Eigen::VectorXd b(numPoints);

  for (int i = 0; i < numPoints; i++) {
    A.row(i) << points[i].x(), points[i].y(), 1.0;
    b(i) = points[i].z();
  }

  // 使用最小二乘方法求解平面参数
  Eigen::VectorXd x = A.colPivHouseholderQr().solve(b);

  // 平面法向量为(x, y, -1)
  V3D normal(x(0), x(1), -1.0);
  normal.normalize();
  return normal;
}

// 使用SVD进行平面拟合
V3D fitPlaneWithSVD(const std::vector<V3D>& points) {
  int numPoints = points.size();

  // 构建数据矩阵
  Eigen::MatrixXd dataMatrix(numPoints, 3);
  for (int i = 0; i < numPoints; i++) {
    dataMatrix.row(i) = points[i].transpose();
  }

  // 去除数据均值
  V3D mean = dataMatrix.colwise().mean();
  dataMatrix.rowwise() -= mean.transpose();

  // 使用奇异值分解求解平面法向量
  Eigen::JacobiSVD<Eigen::MatrixXd> svd(dataMatrix, Eigen::ComputeThinU | Eigen::ComputeFullV);

  V3D normal = svd.matrixV().col(2);
  std::cout << "Singular values of A:" << svd.singularValues().transpose() << std::endl;

  return normal;
}

std::vector<V3D> generateRandomPointCloud(int numPoints, double radius) {
  std::vector<V3D> points;
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<double> dis(0.0, 1.0);

  for (int i = 0; i < numPoints; i++) {
    double theta = 2.0 * M_PI * dis(gen);
    double phi = acos(2.0 * dis(gen) - 1.0);
    double r = radius;

    double x = r * sin(phi) * cos(theta);
    double y = r * sin(phi) * sin(theta);
    double z = r * cos(phi);

    points.emplace_back(x, y, z);
  }
  return points;
}

std::vector<V3D> generatePointCloud(double radius, double pointSpacing) {
  std::vector<V3D> points;
  double surfaceArea = 4.0 * M_PI * radius * radius;
  int numPoints = static_cast<int>(surfaceArea / (pointSpacing * pointSpacing));
  int numPointsPerDim = sqrt(numPoints);
  for (int i = 0; i < numPointsPerDim; i++) {
    for (int j = 1; j < numPointsPerDim; j++) {
      double theta = 2.0 * M_PI * i / numPointsPerDim;
      double phi = M_PI * j / numPointsPerDim;

      double x = radius * sin(phi) * cos(theta);
      double y = radius * sin(phi) * sin(theta);
      double z = radius * cos(phi);

      points.emplace_back(x, y, z);
    }
  }
  points.emplace_back(0, 0, radius);
  points.emplace_back(0, 0, -radius);
  return points;
}

std::vector<V3D> addNoiseToPointCloud(const std::vector<V3D>& points, double noiseStdDev) {
  std::random_device rd;
  std::mt19937 gen(rd());
  std::normal_distribution<double> noiseDist(0.0, noiseStdDev);

  std::vector<V3D> noisyPoints;
  for (const auto& point : points) {
    double xOffset = noiseDist(gen);
    double yOffset = noiseDist(gen);
    double zOffset = noiseDist(gen);

    V3D noisyPoint(point.x() + xOffset, point.y() + yOffset, point.z() + zOffset);
    noisyPoints.push_back(noisyPoint);
  }
  return noisyPoints;
}

TEST(PlaneFittingTest, AccuracyPerformanceTest) {
  double radius = 10;
  double pointSpacing = 0.3;
  std::cout << "radius:" << radius << std::endl;
  std::cout << "pointSpacing:" << pointSpacing << std::endl;
  double surfaceArea = 4.0 * M_PI * radius * radius;
  int numPoints = static_cast<int>(surfaceArea / (pointSpacing * pointSpacing));
  std::vector<V3D> points_vec = generateRandomPointCloud(numPoints, radius);

  double noiseStdDev = 0.01;  // 各个轴都有1cm标准差的噪声
  std::vector<V3D> noisyPoints_vec = addNoiseToPointCloud(points_vec, noiseStdDev);

  PointCloud::Ptr cloud(new PointCloud);

  for (const auto& point : noisyPoints_vec) {
    PointT pclPoint;
    pclPoint.x = point.x();
    pclPoint.y = point.y();
    pclPoint.z = point.z();
    cloud->push_back(pclPoint);
  }
  pcl::KdTreeFLANN<PointT> kdtree;
  kdtree.setInputCloud(cloud);

  for (int k = 5; k <= 20; ++k) {
    std::cout << "k:" << k << std::endl;

    double fitting_ts_PCA = 0;
    double fitting_ts_SVD = 0;
    double fitting_ts_LeastSquare = 0;

    double error_theta_PCA = 0;
    double error_theta_SVD = 0;
    double error_theta_LeastSquare = 0;

    for (int i = 0; i < points_vec.size(); ++i) {
      PointT queryPoint;
      queryPoint.x = points_vec[i].x();  // 假设查询点为原点
      queryPoint.y = points_vec[i].y();
      queryPoint.z = points_vec[i].z();

      std::vector<int> indices(k);
      std::vector<float> distances(k);

      // std::cout << "queryPoint:" << queryPoint << std::endl;
      kdtree.nearestKSearch(queryPoint, k, indices, distances);

      std::vector<V3D> neighbor_vec;
      for (size_t j = 0; j < indices.size(); ++j) {
        int neighborIndex = indices[j];
        const PointT& neighborPoint = cloud->points[neighborIndex];
        V3D neighbor(neighborPoint.x, neighborPoint.y, neighborPoint.z);

        std::cout << "point:" << neighbor.transpose() << std::endl;
        neighbor_vec.push_back(neighbor);
      }

      V3D normalTrue = points_vec[i];

      TicToc SVD;
      V3D normalSVD = fitPlaneWithSVD(neighbor_vec);
      fitting_ts_SVD += SVD.Toc();

      TicToc PCA;
      V3D normalPCA = fitPlaneWithPCA(neighbor_vec);
      fitting_ts_PCA += PCA.Toc();

      TicToc LeastSquare;
      V3D normalLeastSquare = fitPlaneWithLeastSquares(neighbor_vec);
      fitting_ts_LeastSquare += LeastSquare.Toc();

      if (normalSVD.dot(normalTrue) < 0) normalSVD *= -1;
      if (normalPCA.dot(normalTrue) < 0) normalPCA *= -1;
      if (normalLeastSquare.dot(normalTrue) < 0) normalLeastSquare *= -1;

      Eigen::Quaterniond err_svd = Eigen::Quaterniond::FromTwoVectors(normalSVD, normalTrue);
      Eigen::Quaterniond err_pca = Eigen::Quaterniond::FromTwoVectors(normalPCA, normalTrue);
      Eigen::Quaterniond err_leastsquare =
          Eigen::Quaterniond::FromTwoVectors(normalLeastSquare, normalTrue);

      error_theta_SVD += acos(err_svd.w()) * 57.3;
      error_theta_PCA += acos(err_pca.w()) * 57.3;
      error_theta_LeastSquare += acos(err_leastsquare.w()) * 57.3;

      std::cout << "------------------------------------------" << std::endl;
    }

    std::cout << "error_theta_SVD:" << error_theta_SVD / points_vec.size() << std::endl;
    std::cout << "error_theta_PCA:" << error_theta_PCA / points_vec.size() << std::endl;
    std::cout << "error_theta_LeastSquare:" << error_theta_LeastSquare / points_vec.size()
              << std::endl;
    std::cout << "fitting_ts_SVD:" << fitting_ts_SVD << std::endl;
    std::cout << "fitting_ts_PCA:" << fitting_ts_PCA << std::endl;
    std::cout << "fitting_ts_LeastSquare:" << fitting_ts_LeastSquare << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
  }
}
