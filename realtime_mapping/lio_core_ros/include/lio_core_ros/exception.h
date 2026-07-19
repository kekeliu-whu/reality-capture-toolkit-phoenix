#pragma once

//
// if you change the error code,
//   https://pecivkvtit.feishu.cn/docx/IOlTdewQ5oIMpsxpCPqcNNmrnGd must be updated
//

#ifdef __cplusplus
extern "C"
{
#endif

  enum class LioCoreRosErrorCode
  {
    EC_LIO_OK = 0,

    // 入参检查
    // 入参检查
    EC_LIO_UNSUPPORTED_DEVICE_MODEL = 70001,    // 不支持的设备类型
    EC_LIO_OUTPUT_DIR_NOT_EXISTS = 70002,       // 输出目录不存在
    EC_LIO_OUTPUT_DIR_TEMP_NOT_EXISTS = 70003,  // 输出临时目录不存在
    EC_LIO_MISSING_HBC_FILE = 70004,            // HBC文件缺失

    // 输入文件异常
    EC_LIO_CREATE_EXTRACTED_CALIB_FILE = 70005,  // 导出标定文件到临时文件夹失败，L1不会出现该错误
    EC_LIO_MISSING_CALIB_FILE = 70006,           // 缺失标定文件
    EC_LIO_PARSE_CALIB_FILE = 70007,             // 标定文件解析失败
    EC_LIO_PARSE_ALGO_FILE = 70008,              // 算法文件解析失败
    EC_LIO_PARSE_HBC_FILE = 70009,               // HBC文件解析失败
    // 输出文件异常
    EC_LIO_CREATE_XBC_FILE = 70010,      // XBC文件创建失败
    EC_LIO_CREATE_LIO_ULG_FILE = 70011,  // ULOG文件创建失败

    // 传感器数据异常
    EC_LIO_HBC_READ_IMU_FAILED = 70012,       // hbc中读取IMU失败（缺失或时间短于20s）
    EC_LIO_HBC_READ_ENCODER_FAILED = 70013,   // hbc中读取编码器数据失败（缺失或时间短于20s，K1这种没有电机的不会报该错误）
    EC_LIO_HBC_READ_LIDAR_FAILED = 70014,     // hbc中读取雷达数据失败（缺失或时间短于20s）

    EC_LIO_UNKNOWN = 79999
  };

#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
#include <stdexcept>
class LioCoreRosException : public std::runtime_error
{
 public:
  LioCoreRosException(int code) : std::runtime_error("LioCoreRosException"), code_(code)
  {
  }

 private:
  int code_;
};
#endif
