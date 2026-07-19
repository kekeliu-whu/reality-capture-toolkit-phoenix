cmake_minimum_required(VERSION 3.12)
project(lio_core_app)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
add_compile_options($<$<CXX_COMPILER_ID:MSVC>:/MP>)

# 第三方依赖库
find_package(glog CONFIG REQUIRED)
find_package(Boost REQUIRED COMPONENTS thread filesystem serialization)
find_package(Eigen3 REQUIRED)
find_package(libLAS  REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(jsoncpp CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)
find_package(rosbag REQUIRED)
find_package(OpenMP REQUIRED)

add_definitions("-DGLOG_USE_GLOG_EXPORT -DGLOG_USE_GFLAGS -DGLOG_NO_ABBREVIATED_SEVERITIES -DNOMINMAX -DMP_EN")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

message("OpenMP: " ${OpenMP_CXX_FLAGS})

# 编译lio_core
set(LIO_CORE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core)
add_subdirectory(${LIO_CORE_PATH}/thirdparty/GeographicLib-2.1.1/ geometry)
add_subdirectory(${LIO_CORE_PATH}/thirdparty/xmap/ xmap)
add_subdirectory(${LIO_CORE_PATH}/thirdparty/lslog lslog)

add_library(lio_core STATIC
    ${LIO_CORE_PATH}/src/ieskf/ieskf.cpp
    ${LIO_CORE_PATH}/src/ieskf/ieskf_predict.cpp
    ${LIO_CORE_PATH}/src/ieskf/ieskf_update.cpp
    ${LIO_CORE_PATH}/src/ieskf/state_group.cpp
    ${LIO_CORE_PATH}/src/sensor_fusion/lidar_fusion.cpp
    ${LIO_CORE_PATH}/src/sensor_fusion/static_fusion.cpp
    ${LIO_CORE_PATH}/src/sensor_fusion/base_fusion.cpp
    ${LIO_CORE_PATH}/src/lio.cpp
    ${LIO_CORE_PATH}/src/common/common_lib.cpp
    ${LIO_CORE_PATH}/src/method/lio_init.cpp
    ${LIO_CORE_PATH}/src/method/preprocess.cpp
    ${LIO_CORE_PATH}/src/method/io_utils.cpp
    ${LIO_CORE_PATH}/src/method/uniform_sampling.cpp
    ${LIO_CORE_PATH}/src/method/undistort.cpp
)

target_include_directories(lio_core PUBLIC
    ${LIO_CORE_PATH}/interface
    ${LIO_CORE_PATH}/src
    ${LIO_CORE_PATH}/thirdparty/GeographicLib/include
    ${LIO_CORE_PATH}/thirdparty/sophus
    ${LIO_CORE_PATH}/thirdparty/xmap/include
    ${LIO_CORE_PATH}/thirdparty/lslog/include
)

# 编译windows接口库
set(DLL_INTF_PATH  ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/src)
include_directories(
    ${LIO_CORE_PATH}/interface
    ${LIO_CORE_PATH}/src
    ${LIO_CORE_PATH}/thirdparty/xmap/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ros/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ros/pcl_conversions
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/lixel_msgs-f825749
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ros/roscpp_core/rostime/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ros/roscpp_core/roscpp_traits/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ros/roscpp_core/cpp_common/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/lslog/include
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ulog_cpp-591e5f9
    ${CMAKE_CURRENT_SOURCE_DIR}/../lio_core/thirdparty/lslog/include
    ${Boost_INCLUDE_DIRS}
    ${OpenCV_INCLUDE_DIRS}
    ${EIGEN3_INCLUDE_DIRS}
    ${libLAS_INCLUDE_DIRS}
    ${PCL_INCLUDE_DIRS}
    ${YAML_CPP_INCLUDE_DIR}
    ${JSON_INC_PATH}
    ${CLI11_INCLUDE_DIRS}
    ${GeographicLib_INCLUDE_DIRS}
    ${libnabo_INCLUDE_DIRS}
    ${GLOG_INCLUDE_DIRS}
    ${SPDLOG_INCLUDE_DIRS}
)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../lio_core_ros/thirdparty/ulog_cpp-591e5f9 ulog_cpp)

add_library(lio_interface SHARED
    ${DLL_INTF_PATH}/lio_interface.cpp
    ${DLL_INTF_PATH}/subscriber.cpp
    ${DLL_INTF_PATH}/publisher.cpp
    ${DLL_INTF_PATH}/parameters_reader.cpp
    ${DLL_INTF_PATH}/data_io.cpp
    ${DLL_INTF_PATH}/ulog.cpp
)

target_link_libraries(
    lio_interface
    glog::glog
    xmap_lib
    lio_core
    rosbag::rosbag
    yaml-cpp
    ${GLOG_LIBRARYS}
    ulog_cpp
    xgrids::log
    OpenMP::OpenMP_CXX
)

add_executable(
    lio_interface_demo
    ${DLL_INTF_PATH}/lio_interface_demo.cpp
)

target_link_libraries(lio_interface_demo lio_interface)
