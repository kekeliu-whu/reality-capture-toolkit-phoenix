#include <cstdlib>
#include <cstring>

#include <iostream>
#include "lio_core_ros/lio_interface.h"

constexpr int PATH_MAX_LEN = 1024;

int main()
{
  LioStartParam lio_interface;
  lio_interface.device_type = LioDeviceType::LIO_DT_LIXEL_K1;
  lio_interface.bag_filename = (char*)malloc(PATH_MAX_LEN);
  strcpy(
      lio_interface.bag_filename,
      R"(g:\tools\test_mid360\k1-test\2024-03-21-175531.hbc)");
  lio_interface.output_dir = (char*)malloc(PATH_MAX_LEN);
  strcpy(lio_interface.output_dir, R"(g:\tools\test_mid360\k1-test\tmp)");
  lio_interface.output_dir_temp = (char*)malloc(PATH_MAX_LEN);
  strcpy(lio_interface.output_dir_temp, R"(g:\tools\test_mid360\k1-test\tmp)");

  auto lio_handle = LioInterface_Init();
  int code = LioInterface_Start(lio_handle, &lio_interface);
  std::cout << code << std::endl;
  double progress;
  LioInterface_GetProgress(lio_handle, &progress);
  LioInterface_Cleanup(&lio_handle);
}
