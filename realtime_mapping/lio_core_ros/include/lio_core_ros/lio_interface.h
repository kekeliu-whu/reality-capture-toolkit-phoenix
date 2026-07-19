#pragma once

#include "device_types.h"

#ifndef __linux__

#ifdef LIO_DLL_EXPORTS
#define LIO_DLL_API _declspec(dllexport)
#else
#define LIO_DLL_API _declspec(dllimport)
#endif  // LIO_DLL_EXPORTS

#else

#define LIO_DLL_API

#endif

#ifdef __cplusplus
extern "C"
{
#endif

  typedef LIO_DLL_API struct LioStartParam
  {
    LioDeviceType device_type;
    char* bag_filename;     // hbc filename
    char* output_dir;       // save lio.ulog and log files
    char* output_dir_temp;  // save lio.xbc and exported yaml files
  } LioStartParam;

  typedef void* LIOHANDLE;

  LIO_DLL_API LIOHANDLE LioInterface_Init();

  // return ok or error code
  LIO_DLL_API int LioInterface_Start(LIOHANDLE handle, const LioStartParam* param);

  LIO_DLL_API int LioInterface_GetProgress(LIOHANDLE handle, double* progress);

  LIO_DLL_API int LioInterface_Cancel(LIOHANDLE handle);

  LIO_DLL_API int LioInterface_Cleanup(LIOHANDLE* handle);

#ifdef __cplusplus
}
#endif
