#include <Eigen/Dense>  // 包含Eigen库
#include <fstream>
#include <iostream>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace Eigen;
using namespace std;

struct Pose
{
  double ts{};
  Vector3d pose;
  Quaterniond rot;
};

// 将Pose数据写入CSV文件
void writePosesToCSV(const vector<Pose>& poses, const string& filename)
{
  ofstream file(filename);
  if (!file.is_open())
  {
    cerr << "Failed to open the file." << endl;
    return;
  }

  // 逐行写入数据
  for (const auto& pose : poses)
  {
    file << std::fixed << pose.ts << "," << pose.pose.x() << "," << pose.pose.y() << "," << pose.pose.z() << ","
         << pose.rot.w() << "," << pose.rot.x() << "," << pose.rot.y() << "," << pose.rot.z() << endl;
  }

  file.close();
  cout << "CSV file has been written successfully." << endl;
}

int main()
{
  // 打开CSV文件进行读取
  ifstream file_gnss(
      "/media/youyuan/ssd/Datasets/静止rtk/2024-04-10-105101_V2_120m自用RTK放点_f68_V2_120/temp_optimized_map/"
      "gnss_xyz.csv");
  vector<Pose> gnss;  // 用于存储Pose数据
  string line;

  // 逐行读取文件内容
  while (getline(file_gnss, line))
  {
    istringstream ss(line);
    Pose pose;
    string cell;
    // 逐个解析每个单元格的数据
    getline(ss, cell, ' ');  // 读取时间戳
    pose.ts = stod(cell);
    for (int i = 0; i < 3; ++i)
    {
      getline(ss, cell, ' ');  // 读取x、y、z坐标
      pose.pose(i) = stod(cell);
    }
    // z轴减去天线
    pose.pose(2) = pose.pose(2) - 0.0594;
    gnss.push_back(pose);  // 将当前Pose添加到poses中
  }

  cout << "gnss.size:" << gnss.size() << std::endl;
  writePosesToCSV(
      gnss,
      "/media/youyuan/ssd/Datasets/静止rtk/2024-04-10-105101_V2_120m自用RTK放点_f68_V2_120/temp_optimized_map/"
      "gnss_ground.csv");
  return 0;
}