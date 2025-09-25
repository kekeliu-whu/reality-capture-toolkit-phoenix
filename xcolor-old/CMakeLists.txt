cmake_minimum_required(VERSION 3.10)
project(xcolor)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

if(OPT_ASAN)
  add_definitions(-D_DISABLE_VECTOR_ANNOTATION)
  add_definitions(-D_DISABLE_STRING_ANNOTATION)
  add_compile_options(/fsanitize=address)
endif()

if(MSVC)
  add_compile_options(/MP)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/MD /Zi /O2 /Ob3")
endif()

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DGLOG_USE_GLOG_EXPORT)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(gflags REQUIRED)
find_package(PCL REQUIRED)
find_package(Boost COMPONENTS program_options)
find_package(colmap REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(mimalloc CONFIG REQUIRED)
find_package(yaml-cpp CONFIG REQUIRED)
find_package(tinyxml CONFIG REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(PROJ CONFIG REQUIRED)

include_directories(
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

set(
  PGO_PRIVATE_SRCS
  src/common/histogram.cc
  src/core/xcolor_lib.cc
  src/core/xsfm_lib.cc
  src/core/utils.cc
  src/core/run_ba.cc
  src/io/colmap_io.cc
  src/io/xml_io.cc
  src/io/io_utils.cc
  src/common/histogram.h
  src/common/types.h
  src/core/run_ba.h
  src/core/sfm_config.h
  src/core/utils.h
  src/core/xcolor_lib.h
  src/core/xsfm_lib.h
  src/io/colmap_io.h
  src/io/io_utils.h
  src/io/xml_io.h
  src/tools/internal/camera.h
  src/tools/internal/poly_fisheye_camera.h
  src/tools/internal/utils/cv_utils.h
  src/tools/internal/utils/eigen_utils.h
  src/tools/internal/utils/gpl.h
  src/tools/internal/utils/math_utils.h
  src/tools/internal/utils/polynomial.h
  src/tools/internal/utils/ros_utils.h
  src/tools/internal/utils/sys_utils.h
)

set(PGO_PRIVATE_LIBS
  mimalloc
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  colmap::colmap
  absl::flat_hash_map
  absl::flat_hash_set
  ${OpenCV_LIBS}
  rapidjson
  yaml-cpp::yaml-cpp
  unofficial-tinyxml::unofficial-tinyxml
  $<TARGET_NAME:pdalcpp>
  PROJ::proj
)

function(add_private_tool name)
  add_executable(${name} ${ARGN} ${PGO_PRIVATE_SRCS})
  target_link_libraries(${name} ${PGO_PRIVATE_LIBS})
endfunction()

add_private_tool(xcolor_main src/xcolor_main.cc)
add_private_tool(xsfm src/xsfm_main.cc)
add_private_tool(xsfm_reset_s20_cameras src/tools/xsfm_reset_s20_cameras.cc)
add_private_tool(process_point_cloud_s20 src/tools/process_point_cloud_s20.cc)
add_private_tool(xsfm_undistort_s20_images
  src/tools/xsfm_undistort_s20_images.cc
  src/tools/internal/poly_fisheye_camera.cc
  src/tools/internal/camera.cc
  src/tools/internal/utils/cv_utils.cc
  src/tools/internal/utils/polynomial.cc
  src/tools/internal/utils/gpl.cc
)
