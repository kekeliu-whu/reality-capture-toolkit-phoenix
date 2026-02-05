cmake_minimum_required(VERSION 3.10)
project(vikit VERSION 1.0)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(OpenCV CONFIG REQUIRED)

include_directories(
  ${OPENCV_INCLUDE_DIRS}
)

add_library(${PROJECT_NAME} STATIC
  src/atan_camera.cpp
  src/equidistant_camera.cpp
  src/homography.cpp
  src/img_align.cpp
  src/math_utils.cpp
  src/omni_camera.cpp
  src/performance_monitor.cpp
  src/pinhole_camera.cpp
  src/polynomial_camera.cpp
  src/robust_cost.cpp
  src/vision.cpp
)

target_include_directories(${PROJECT_NAME} PUBLIC
  include
)
target_link_libraries(
  ${PROJECT_NAME}
  ${OpenCV_LIBS}
  sophus
)
