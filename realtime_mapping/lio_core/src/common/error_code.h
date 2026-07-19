#pragma once
#include <map>
#include <string>
#define moduleIDLeftMove  24u
#define functionIDLeftMove 16u

typedef uint8_t ModuleIDType;
typedef uint8_t FunctionIDType;
typedef uint8_t ErrorLevelType;
typedef uint8_t RawRetCodeType;
typedef uint32_t ErrorCodeType;

enum ErrorLevel : ErrorLevelType
{
  INFO = 0,
  WARNING = 1,
  LIO_ERROR = 2,
  FATAL = 3
};


static constexpr ErrorCodeType
ErrorCode(ModuleIDType module_id, FunctionIDType function_id, RawRetCodeType error_code)
{
  return (
      ((ErrorCodeType)module_id << moduleIDLeftMove) |
      ((ErrorCodeType)function_id << functionIDLeftMove) | (ErrorCodeType)error_code);
}

class Error
{
  public:
  const static ErrorCodeType SUCCESS = 0;

  private:
  // Note: all module's index start from 1
  const static ModuleIDType ROS_WRAPPER       = 1;
  const static ModuleIDType LIO_CORE          = 2;
  const static ModuleIDType MAP_REFINEMENT    = 3;
  const static ModuleIDType LOCAL_OPTMIZATION = 4;
  const static ModuleIDType PGO_OPTIMIZATION  = 5;

  public:
  class LioCore
  {
    private:
    // Note: all function's index start from 1
    const static FunctionIDType PREPROCESS = 1;
    const static FunctionIDType PREDICT    = 2;
    const static FunctionIDType UNDISTORT  = 3;
    const static FunctionIDType DOWNSAMPLE = 4;
    const static FunctionIDType JACOBI     = 5;
    const static FunctionIDType ESKF       = 6;
    public:
    // Note: all raw error code start from 1
    enum Undistort : ErrorCodeType
    {
      RAW_POINT_CLOUD_IS_EMPTY     = ErrorCode(LIO_CORE, PREPROCESS, 1),
      PREDICT_STATE_IS_EMPTY       = ErrorCode(LIO_CORE, PREPROCESS, 2),
      INVALID_UNDISTORT_TIME_RANGE = ErrorCode(LIO_CORE, PREPROCESS, 3),
    };
  };

};

