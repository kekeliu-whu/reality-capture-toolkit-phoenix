#pragma once

#ifdef __cplusplus
extern "C"
{
#endif

  typedef enum LioDeviceType
  {
    LIO_DT_LIXEL_V1 = 0,         // L1
    LIO_DT_LIXEL_V2 = 1,         // L2
    LIO_DT_LIXEL_M1 = 2,         // 煤安           // l2.yaml
    LIO_DT_LIXEL_K1 = 3,         // 小雷达         // k1.yaml
    LIO_DT_LIXEL_S1 = 4,         // 机载小雷达     // not supported
    LIO_DT_LIXEL_L2PRO = 5,      // L2 pro        // l2_pro.yaml
    LIO_DT_LIXEL_L2PRO_ZHD = 6,  // 中海达        // not supported
    LIO_DT_UNKNOWN = 1000
  } LioDeviceType;

#ifdef __cplusplus
}
#endif
