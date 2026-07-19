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
std::unordered_map<int, Vector3d> readDataFromFile(const std::string& file_name)
{
  std::ifstream file(file_name);              // 打开文件
  std::unordered_map<int, Vector3d> dataMap;  // 用于存储数据的unordered_map
  if (!file.is_open())
  {                                           // 检查文件是否成功打开
    std::cerr << "Error opening file." << std::endl;
    return dataMap;                           // 返回空的unordered_map
  }
  std::string line;
  while (std::getline(file, line))
  {  // 逐行读取文件内容
    std::istringstream iss(line);
    std::string token;
    Eigen::Vector3d xyz;
    // 读取label
    std::getline(iss, token, ',');
    int label = std::stoi(token);
    // 读取x
    std::getline(iss, token, ',');
    double x = std::stod(token);
    // 读取y
    std::getline(iss, token, ',');
    double y = std::stod(token);
    // 读取z
    std::getline(iss, token);
    double z = std::stod(token);
    // 创建Vector3d对象
    xyz = Vector3d(x, y, z);
    // 将数据点添加到unordered_map中
    dataMap[label] = xyz;
  }
  return dataMap;  // 返回包含数据的unordered_map
}

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
  std::string file1("/home/youyuan/Datasets_pc/shanghaiyl真值.csv");   // 替换为你的CSV文件路径
  std::string file2("/home/youyuan/Datasets_pc/rtk_trans_to_cp.csv");  // 替换为你的CSV文件路径

  std::unordered_map<int, Vector3d> dataMap1 = readDataFromFile(file1);
  std::unordered_map<int, Vector3d> dataMap2 = readDataFromFile(file2);

  for (auto& pair : dataMap1)
  {
    int key = pair.first;
    Vector3d value1 = pair.second;
    Vector3d value2 = dataMap2[key];

    for (auto& tgt_pair : dataMap1)
    {
      int tgt_key = tgt_pair.first;
      if (tgt_key == key)
        continue;

      Vector3d tgt_value1 = tgt_pair.second;
      Vector3d tgt_value2 = dataMap2[tgt_key];

      Vector3d diff = (tgt_value1 - value1) - (tgt_value2 - value2);
      std::cout << "pair:" << key << "," << tgt_key << "; diff:" << diff.transpose() << std::endl;
    }
  }

  // 打开CSV文件进行读取
  //  ifstream file("/home/youyuan/Datasets_pc/2024-04-11-151713/temp_optimized_map_rtk_fusion/pose_no_offset.csv");
  //  ifstream file_gnss("/home/youyuan/Datasets_pc/2024-04-11-151713/temp_optimized_map_rtk_fusion/gnss_xyz.csv");
  //  if (!file.is_open())
  //  {
  //    cerr << "Failed to open the file." << endl;
  //    return 1;
  //  }
  //
  //  vector<Pose> poses;  // 用于存储Pose数据
  //  vector<Pose> gnss;   // 用于存储Pose数据
  //  string line;
  //
  //  // 逐行读取文件内容
  //  while (getline(file, line))
  //  {
  //    istringstream ss(line);
  //    Pose pose;
  //    string cell;
  //    // 逐个解析每个单元格的数据
  //    getline(ss, cell, ' ');  // 读取时间戳
  //    pose.ts = stod(cell);
  //    for (int i = 0; i < 3; ++i)
  //    {
  //      getline(ss, cell, ' ');  // 读取x、y、z坐标
  //      pose.pose(i) = stod(cell);
  //    }
  //    for (int i = 0; i < 4; ++i)
  //    {
  //      getline(ss, cell, ' ');  // 读取四元数的四个分量
  //      if (i == 0)
  //        pose.rot.w() = stod(cell);
  //      else if (i == 1)
  //        pose.rot.x() = stod(cell);
  //      else if (i == 2)
  //        pose.rot.y() = stod(cell);
  //      else if (i == 3)
  //        pose.rot.z() = stod(cell);
  //    }
  //    poses.push_back(pose);  // 将当前Pose添加到poses中
  //  }
  //
  //  // 逐行读取文件内容
  //  while (getline(file_gnss, line))
  //  {
  //    istringstream ss(line);
  //    Pose pose;
  //    string cell;
  //    // 逐个解析每个单元格的数据
  //    getline(ss, cell, ' ');  // 读取时间戳
  //    pose.ts = stod(cell);
  //    for (int i = 0; i < 3; ++i)
  //    {
  //      getline(ss, cell, ' ');  // 读取x、y、z坐标
  //      pose.pose(i) = stod(cell);
  //    }
  //    gnss.push_back(pose);  // 将当前Pose添加到poses中
  //  }
  //
  //  Vector3d rtk2imu(-0.00882, -0.06, -0.35109);
  //  Vector3d imu2cp(0.02243, -0.11746, -0.2094);
  //  Vector3d rtk2cp = rtk2imu + imu2cp;
  //
  //  cout << "gnss.size:" << gnss.size() << std::endl;
  //  cout << "poses.size:" << poses.size() << std::endl;
  //
  //  for (int i = 0; i < gnss.size(); ++i)
  //  {
  //    for (int j = 0; j < poses.size() - 1; ++j)
  //    {
  //      if (poses[j].ts < gnss[i].ts && gnss[i].ts < poses[j + 1].ts)
  //      {
  //        double scale = (gnss[i].ts - poses[j].ts) / (poses[j + 1].ts - poses[j].ts);
  //        Quaterniond inter_q = poses[j].rot.slerp(scale, poses[j + 1].rot);
  //        gnss[i].pose = inter_q.toRotationMatrix() * rtk2cp + gnss[i].pose;
  //      }
  //    }
  //  }
  //  writePosesToCSV(gnss, "/home/youyuan/Datasets_pc/2024-04-11-151713/temp_optimized_map_rtk_fusion/gnss2cp.csv");
  return 0;
}