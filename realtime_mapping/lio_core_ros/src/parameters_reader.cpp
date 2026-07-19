#include <rosbag/bag.h>
#include <rosbag/view.h>
#include <yaml-cpp/yaml.h>
#include <boost/filesystem.hpp>
#include <fstream>

#include "lio_core_ros/exception.h"
#include "lio_core_ros/parameters_reader.h"
#include "lixel_msgs/AnyData.h"
#include "log/lsLogger.h"

#define LIO_EC_IF_FAILED(cond, code) \
  if (!(cond))                       \
  {                                  \
    lslog(LSLOG_ERROR) << #cond;           \
    return code;                     \
  }

namespace
{

template <typename Derived>
void ForceToReadEigenMatrix(const YAML::Node& node, Eigen::MatrixBase<Derived>& m, bool check_transform = false)
{
  auto arr = node.as<std::vector<double>>();
  // DCHECK_EQ(arr.size(), m.rows() * m.cols()) << "Read matrix failed because dim mismatch.";
  m = Eigen::Map<const Eigen::Matrix<
      double,
      Derived::RowsAtCompileTime,
      Derived::ColsAtCompileTime,
      Derived::ColsAtCompileTime == 1 ? Eigen::ColMajor : Eigen::RowMajor>>{arr.data()};

  if (check_transform && ((m.rows() == 4 && m.cols() == 4) || (m.rows() == 3 && m.cols() == 3)))
  {
    if (fabs(m.determinant() - 1) > 1e-3)
    {
      lslog(LSLOG_WARNING) << "The matrix is not a rigid transform matrix.";
    }
  }
}

LioCoreRosErrorCode ReadInstrinsicLidar(const std::string& filename, lixel::SensorParam::LidarParam& param)
{
  param.enabled = false;
  if (!boost::filesystem::exists(filename))
  {
    lslog(LSLOG_ERROR) << "File not found: " << filename;
    return LioCoreRosErrorCode::EC_LIO_MISSING_CALIB_FILE;
  }

  try
  {
    YAML::Node config = YAML::LoadFile(filename);

    Eigen::Vector2d s_vec;
    ForceToReadEigenMatrix(config["s"], s_vec);
    param.elevation_offset = config["e"].as<double>();
    param.dist = s_vec[0];
    param.angle = s_vec[1];

    if (param.elevation_offset > 1.0 / 57.3)
    {
      lslog(LSLOG_WARNING) << "lidar calib param invalid, e is larger than 1.0 deg, disable lidar calib";
    }
  }
  catch (const YAML::Exception& e)
  {
    lslog(LSLOG_ERROR) << "BadFile: " << e.what();
    return LioCoreRosErrorCode::EC_LIO_PARSE_CALIB_FILE;
  }

  param.enabled = true;
  return LioCoreRosErrorCode::EC_LIO_OK;
}

LioCoreRosErrorCode ReadInstrinsicImu(const std::string& filename, lixel::SensorParam::ImuParam& param)
{
  param.enabled = false;
  if (!boost::filesystem::exists(filename))
  {
    lslog(LSLOG_ERROR) << "File not found: " << filename;
    return LioCoreRosErrorCode::EC_LIO_MISSING_CALIB_FILE;
  }

  try
  {
    YAML::Node config = YAML::LoadFile(filename);

    ForceToReadEigenMatrix(config["Ta"], param.Ta);
    ForceToReadEigenMatrix(config["Ka"], param.Ka);
    ForceToReadEigenMatrix(config["Ba"], param.Ba);
    ForceToReadEigenMatrix(config["Tg"], param.Tg);
    ForceToReadEigenMatrix(config["Kg"], param.Kg);
    ForceToReadEigenMatrix(config["Bg"], param.Bg);
  }
  catch (const YAML::Exception& e)
  {
    lslog(LSLOG_ERROR) << "BadFile: " << e.what();
    return LioCoreRosErrorCode::EC_LIO_PARSE_CALIB_FILE;
  }

  param.enabled = true;
  return LioCoreRosErrorCode::EC_LIO_OK;
}

LioCoreRosErrorCode ReadExtrinsicFileWithTransform(const std::string& filename, Eigen::Matrix4d& param)
{
  if (!boost::filesystem::exists(filename))
  {
    lslog(LSLOG_ERROR) << "File not found: " << filename;
    return LioCoreRosErrorCode::EC_LIO_MISSING_CALIB_FILE;
  }

  try
  {
    YAML::Node config = YAML::LoadFile(filename);
    ForceToReadEigenMatrix(config["transform"], param, true);
  }
  catch (const YAML::Exception& e)
  {
    lslog(LSLOG_ERROR) << "BadFile: " << e.what();
    return LioCoreRosErrorCode::EC_LIO_PARSE_CALIB_FILE;
  }

  return LioCoreRosErrorCode::EC_LIO_OK;
}

void PrintParams(const FullParameters& params)
{
#define PRINT(x) lslog(LSLOG_INFO) << #x ": " << x

  // print sensor_param
  PRINT(params.sensor_param.lidar_param.enabled);
  PRINT(params.sensor_param.lidar_param.dist);
  PRINT(params.sensor_param.lidar_param.angle);
  PRINT(params.sensor_param.lidar_param.elevation_offset);
  PRINT(params.sensor_param.imu_param.enabled);
  PRINT(params.sensor_param.imu_param.Ta);
  PRINT(params.sensor_param.imu_param.Ka);
  PRINT(params.sensor_param.imu_param.Ba);
  PRINT(params.sensor_param.imu_param.Tg);
  PRINT(params.sensor_param.imu_param.Kg);
  PRINT(params.sensor_param.imu_param.Bg);

  // print extrinsic_param
  PRINT(params.extrinsic_param.motor_param.enabled);
  PRINT(params.extrinsic_param.motor_param.ext_motor_lidar);
  PRINT(params.extrinsic_param.ext_imu_motor);
  PRINT(params.extrinsic_param.t_imu_gnss);

  // print ros_param
  PRINT(params.ros_param.imu_topic);
  PRINT(params.ros_param.encoder_topic);
  PRINT(params.ros_param.lidar_topic);
  PRINT(params.ros_param.gnss_topic);
  PRINT(params.ros_param.offline_mode_start_frame_id);
  PRINT(params.ros_param.offline_mode_end_frame_id);
  PRINT(params.ros_param.offline_mode_speed_ratio);

  // print preprocess_param
  PRINT(params.preprocess_param.range_min);
  PRINT(params.preprocess_param.range_max);
  PRINT(params.preprocess_param.body_mask_min);
  PRINT(params.preprocess_param.body_mask_max);
  PRINT(params.preprocess_param.sweep_duration);
  PRINT(params.preprocess_param.sweep_cut_auto);

  // print kf param
  PRINT(params.kf_param.acc_std);
  PRINT(params.kf_param.acc_bias_std);
  PRINT(params.kf_param.gyr_std);
  PRINT(params.kf_param.gyr_bias_std);
  PRINT(params.kf_param.use_gnss);
  PRINT(params.kf_param.use_vio);
  PRINT(params.kf_param.use_edge);
}

}  // namespace

LioCoreRosErrorCode ReadParameters(
    const std::string& algorithm_config_filename,
    const std::string& external_calib_filepath,
    FullParameters& params,
    CalibModel calib_model)
{
  LioCoreRosErrorCode code;
  if ((code = ReadCalibParameters(external_calib_filepath, params, calib_model)) != LioCoreRosErrorCode::EC_LIO_OK)
  {
    return code;
  }
  if ((code = ReadLioParameters(algorithm_config_filename, params)) != LioCoreRosErrorCode::EC_LIO_OK)
  {
    return code;
  }

  PrintParams(params);

  return LioCoreRosErrorCode::EC_LIO_OK;
}

LioCoreRosErrorCode
ReadCalibParameters(const std::string& external_calib_filepath, FullParameters& params, CalibModel calib_model)
{
  LioCoreRosErrorCode code;
  if (calib_model == CalibModel::L2_LIKE)
  {
    if ((code = ReadInstrinsicImu(external_calib_filepath + "/imu.yaml", params.sensor_param.imu_param)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
    if ((code = ReadInstrinsicLidar(external_calib_filepath + "/lidar.yaml", params.sensor_param.lidar_param)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
    if ((code = ReadExtrinsicFileWithTransform(
             external_calib_filepath + "/extrinsic_motor_lidar.yaml",
             params.extrinsic_param.motor_param.ext_motor_lidar)) != LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
    params.extrinsic_param.motor_param.enabled = true;
    if ((code = ReadExtrinsicFileWithTransform(
             external_calib_filepath + "/extrinsic_imu_motor.yaml", params.extrinsic_param.ext_imu_motor)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
  }
  else if (calib_model == CalibModel::K1)
  {
    if ((code = ReadInstrinsicImu(external_calib_filepath + "/imu.yaml", params.sensor_param.imu_param)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
    if ((code = ReadExtrinsicFileWithTransform(
             external_calib_filepath + "/extrinsic_imu_lidar.yaml", params.extrinsic_param.ext_imu_motor)) !=
        LioCoreRosErrorCode::EC_LIO_OK)
    {
      return code;
    }
  }
  else
  {
    lslog(LSLOG_ERROR) << "impossible to be here";
    return LioCoreRosErrorCode::EC_LIO_UNKNOWN;
  }
  return LioCoreRosErrorCode::EC_LIO_OK;
}

LioCoreRosErrorCode ReadLioParameters(const std::string& filename, FullParameters& params)
{
  if (!boost::filesystem::exists(filename))
  {
    lslog(LSLOG_ERROR) << "File not found: " << filename;
    return LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE;
  }

  try
  {
    YAML::Node config = YAML::LoadFile(filename);

    // load ros_param
    LIO_EC_IF_FAILED(config["ros"]["imu_topic"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["ros"]["encoder_topic"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["ros"]["lidar_topic"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["ros"]["gnss_topic"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["ros"]["offline_mode_start_frame_id"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["ros"]["offline_mode_end_frame_id"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["ros"]["offline_mode_speed_ratio"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    params.ros_param.imu_topic = config["ros"]["imu_topic"].as<std::string>();
    params.ros_param.encoder_topic = config["ros"]["encoder_topic"].as<std::string>();
    params.ros_param.lidar_topic = config["ros"]["lidar_topic"].as<std::string>();
    params.ros_param.gnss_topic = config["ros"]["gnss_topic"].as<std::string>();
    params.ros_param.offline_mode_start_frame_id = config["ros"]["offline_mode_start_frame_id"].as<int>();
    params.ros_param.offline_mode_end_frame_id = config["ros"]["offline_mode_end_frame_id"].as<int>();
    params.ros_param.offline_mode_speed_ratio = config["ros"]["offline_mode_speed_ratio"].as<int>();

    // load preprocess_param
    LIO_EC_IF_FAILED(config["preprocess"]["range_min"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["preprocess"]["range_max"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["preprocess"]["body_mask_min"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["preprocess"]["body_mask_max"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["preprocess"]["sweep_duration"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["preprocess"]["sweep_cut_auto"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    params.preprocess_param.range_min = config["preprocess"]["range_min"].as<double>();
    params.preprocess_param.range_max = config["preprocess"]["range_max"].as<double>();
    ForceToReadEigenMatrix(config["preprocess"]["body_mask_min"], params.preprocess_param.body_mask_min);
    ForceToReadEigenMatrix(config["preprocess"]["body_mask_max"], params.preprocess_param.body_mask_max);
    params.preprocess_param.sweep_duration = config["preprocess"]["sweep_duration"].as<double>();
    params.preprocess_param.sweep_cut_auto = config["preprocess"]["sweep_cut_auto"].as<bool>();
    // load downsample_param

    LIO_EC_IF_FAILED(config["downsample"]["area_method"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["downsample"]["init_pca_downsample_dis"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["downsample"]["base_downsample_dis"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["downsample"]["max_downsample_dis"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["downsample"]["ref_downsample_point_num"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);

    params.downsample_param.area_method = lixel::SufaceAreaMethod(config["downsample"]["area_method"].as<int>());
    params.downsample_param.init_pca_downsample_dis = config["downsample"]["init_pca_downsample_dis"].as<float>();
    params.downsample_param.base_downsample_dis = config["downsample"]["base_downsample_dis"].as<float>();
    params.downsample_param.max_downsample_dis = config["downsample"]["max_downsample_dis"].as<float>();
    params.downsample_param.ref_downsample_point_num = config["downsample"]["ref_downsample_point_num"].as<uint32_t>();

    // load load init_param
    LIO_EC_IF_FAILED(config["init_param"]["init_pos_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["init_param"]["init_vel_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["init_param"]["init_rot_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["init_param"]["init_acc_bias_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(
        config["init_param"]["init_gyro_bias_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);

    params.init_param.init_pos_std = config["init_param"]["init_pos_std"].as<double>();
    params.init_param.init_vel_std = config["init_param"]["init_vel_std"].as<double>();
    params.init_param.init_rot_std = config["init_param"]["init_rot_std"].as<double>();
    params.init_param.init_acc_bias_std = config["init_param"]["init_acc_bias_std"].as<double>();
    params.init_param.init_gyro_bias_std = config["init_param"]["init_gyro_bias_std"].as<double>();

    // load kf_param
    LIO_EC_IF_FAILED(config["kf"]["acc_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["acc_bias_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["gyr_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["gyr_bias_std"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["max_iter"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["acc_keep_std_limit"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["gyro_keep_std_limit"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["use_gnss"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["use_vio"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["use_edge"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);
    LIO_EC_IF_FAILED(config["kf"]["k_for_adaptive_search"].IsDefined(), LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE);

    params.kf_param.acc_std = config["kf"]["acc_std"].as<double>();
    params.kf_param.acc_bias_std = config["kf"]["acc_bias_std"].as<double>();
    params.kf_param.gyr_std = config["kf"]["gyr_std"].as<double>();
    params.kf_param.gyr_bias_std = config["kf"]["gyr_bias_std"].as<double>();
    params.kf_param.max_iter = config["kf"]["max_iter"].as<int>();
    params.kf_param.acc_keep_std_limit = config["kf"]["acc_keep_std_limit"].as<double>();
    params.kf_param.gyro_keep_std_limit = config["kf"]["gyro_keep_std_limit"].as<double>();
    params.kf_param.use_gnss = config["kf"]["use_gnss"].as<bool>();
    params.kf_param.use_vio = config["kf"]["use_vio"].as<bool>();
    params.kf_param.use_edge = config["kf"]["use_edge"].as<bool>();
    params.kf_param.k_for_adaptive_search = config["kf"]["k_for_adaptive_search"].as<double>();
  }
  catch (const YAML::Exception& e)
  {
    lslog(LSLOG_ERROR) << "BadFile: " << e.what();
    return LioCoreRosErrorCode::EC_LIO_PARSE_ALGO_FILE;
  }
  return LioCoreRosErrorCode::EC_LIO_OK;
}

std::string GetRealYamlFilename(const std::string& path)
{
  // Find the position of the last '/' character
  std::size_t last_slash_pos = path.rfind('/');

  // If '/' is found, extract the substring after it; otherwise, use the entire path
  std::string file_name = (last_slash_pos != std::string::npos) ? path.substr(last_slash_pos + 1) : path;

  // Find the position of the first '_' character
  std::size_t first_underscore_pos = file_name.rfind('_');

  // If '_' is found, extract the substring before it; otherwise, use the whole file name
  file_name = (first_underscore_pos != std::string::npos) ? file_name.substr(0, first_underscore_pos) : file_name;

  // Add .yaml suffix
  file_name += ".yaml";

  return file_name;
}

bool WriteToFile(const std::string& filename, const std::string& content)
{
  std::ofstream outfile(filename);

  if (!outfile.is_open())
  {
    return false;
  }

  outfile << content;

  return (bool)outfile;
}

int ExtractYamlFileFromHbc(const std::string& bag_filename, const std::string& external_calib_filepath)
{
  int yaml_file_count = 0;

  rosbag::Bag bag;
  bag.open(bag_filename, rosbag::bagmode::Read);
  for (rosbag::MessageInstance const m : rosbag::View(bag))
  {
    auto msg = m.instantiate<lixel_msgs::AnyData>();
    if (!msg)
    {
      continue;
    }

    std::string topic = m.getTopic();
    if (topic.find("_yaml") != std::string::npos)
    {
      std::string real_yaml_filename = GetRealYamlFilename(topic);
      std::string content = msg->data;
      if (WriteToFile(external_calib_filepath + "/" + real_yaml_filename, content))
      {
        lslog(LSLOG_INFO) << "Write yaml file " << external_calib_filepath + "/" + real_yaml_filename << " successfully.";
        yaml_file_count++;
      }
      else
      {
        lslog(LSLOG_ERROR) << "Write yaml file " << external_calib_filepath + "/" + real_yaml_filename << " failed.";
      }
    }
  }

  bag.close();

  return yaml_file_count;
}
