cmake_minimum_required(VERSION 3.0.2)
project(voxel_slam)

set(CMAKE_BUILD_TYPE "Release")
# set(CMAKE_CXX_FLAGS "-std=c++14")
# set(CMAKE_CXX_FLAGS_RELEASE "-O3 -Wall -g")

ADD_COMPILE_OPTIONS(-std=c++14 )
set(CMAKE_CXX_FLAGS "-std=c++14 -O3" )

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions" )
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++14 -pthread -std=c++0x -std=c++14 -fexceptions")

find_package(catkin REQUIRED COMPONENTS
  pcl_conversions
  livox_ros_driver2
  sensor_msgs
  rosbag
)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
# GTSAM cmake config has broken CppUnitLite dependency - find manually
set(GTSAM_INCLUDE_DIR "/usr/include")
find_library(GTSAM_LIBRARY gtsam PATHS /usr/lib/x86_64-linux-gnu REQUIRED)
message(STATUS "GTSAM manually configured: ${GTSAM_LIBRARY}")
# find_package(Ceres REQUIRED)

catkin_package(
  CATKIN_DEPENDS sensor_msgs pcl_conversions livox_ros_driver2 rosbag
  DEPENDS EIGEN3 PCL GTSAM
  INCLUDE_DIRS
)

include_directories(
  include
  ${catkin_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
# ${CERES_INCLUDE_DIRS}
  ${GTSAM_INCLUDE_DIR}
)

add_executable(voxelslam src/voxelslam.cpp src/BTC.cpp)
target_link_libraries(voxelslam ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${GTSAM_LIBRARY} yaml-cpp)







