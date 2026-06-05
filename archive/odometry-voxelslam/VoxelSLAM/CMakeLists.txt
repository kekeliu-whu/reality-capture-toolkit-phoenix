cmake_minimum_required(VERSION 3.12)
project(voxel_slam)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

if(MSVC)
  add_compile_options(/MP /bigobj)
  # Suppress warnings from third-party ROS code patterns
  add_compile_options(/wd4244 /wd4267 /wd4305 /wd4996)
  # Needed for M_PI etc. (scoped per-target below)
endif()

# ===== Dependencies via vcpkg =====
find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED COMPONENTS common io kdtree)
find_package(yaml-cpp REQUIRED)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
find_package(OpenCV REQUIRED)
find_package(GTSAM REQUIRED)

# ===== Migration rosbag (pre-built static lib) =====
set(MIGRATION_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../migration")
set(ROSBAG_INCLUDE_DIR
    "${MIGRATION_ROOT}/thirdparty/ros/include"
    "${MIGRATION_ROOT}/thirdparty/ros/console_bridge/include"
    "${MIGRATION_ROOT}/thirdparty/ros/roscpp_core/cpp_common/include"
    "${MIGRATION_ROOT}/thirdparty/ros/roscpp_core/rostime/include"
    "${MIGRATION_ROOT}/thirdparty/ros/roscpp_core/roscpp_serialization/include"
    "${MIGRATION_ROOT}/thirdparty/ros/roscpp_core/roscpp_traits/include"
    "${MIGRATION_ROOT}/thirdparty/ros/ros_comm/utilities/roslz4/include"
    "${MIGRATION_ROOT}/thirdparty/ros/ros_comm/tools/rosconsole/include"
    "${MIGRATION_ROOT}/thirdparty/ros/ros_comm/tools/rosbag_storage/include"
    "${MIGRATION_ROOT}/thirdparty/ros/vision_opencv-noetic/cv_bridge/include"
    "${MIGRATION_ROOT}/thirdparty/ros/vision_opencv-noetic/cv_bridge/src"
)

find_library(ROSBAG_LIBRARY rosbag
  PATHS "${MIGRATION_ROOT}/build/RelWithDebInfo"
  NO_DEFAULT_PATH
)
if(NOT ROSBAG_LIBRARY)
  message(FATAL_ERROR "rosbag.lib not found in migration/build/RelWithDebInfo. Build migration first (cd migration && cmake --build build --config RelWithDebInfo).")
endif()
message(STATUS "Found rosbag: ${ROSBAG_LIBRARY}")

# ===== Sources =====
set(VOXELSLAM_SRCS
  src/voxelslam.cpp
  src/BTC.cpp
)

# ===== Headers (for IDE visibility) =====
set(VOXELSLAM_HDRS
  src/voxelslam.hpp
  src/tools.hpp
  src/ekf_imu.hpp
  src/voxel_map.hpp
  src/feature_point.hpp
  src/loop_refine.hpp
  src/preintegration.hpp
  src/BTC.h
  src/platform_compat.hpp
  src/livox_ros_driver2/CustomMsg.h
  src/livox_ros_driver2/CustomPoint.h
)

# ===== Executable =====
add_executable(voxelslam ${VOXELSLAM_SRCS} ${VOXELSLAM_HDRS})

# Patched GTSAM includes to work around MSVC / boost::variant issues
# (This needs to be before other includes so it shadows vcpkg's copy)
set(PATCHED_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/patched_includes)

target_include_directories(voxelslam PRIVATE
  ${PATCHED_INCLUDE_DIR}
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${EIGEN3_INCLUDE_DIR}
  ${PCL_INCLUDE_DIRS}
  ${Boost_INCLUDE_DIRS}
  ${OpenCV_INCLUDE_DIRS}
  ${ROSBAG_INCLUDE_DIR}
)

find_library(BZIP2_LIBRARY bz2)
if(NOT BZIP2_LIBRARY)
  message(FATAL_ERROR "bz2.lib not found. Install bzip2 via vcpkg: vcpkg install bzip2")
endif()
message(STATUS "Found bz2: ${BZIP2_LIBRARY}")

target_link_libraries(voxelslam PRIVATE
  ${ROSBAG_LIBRARY}
  ${PCL_LIBRARIES}
  ${Boost_LIBRARIES}
  yaml-cpp::yaml-cpp
  gtsam
  ${OpenCV_LIBS}
  ${BZIP2_LIBRARY}
)

target_compile_definitions(voxelslam PRIVATE _USE_MATH_DEFINES NOMINMAX)

# ===== Copy DLLs for running =====
if(MSVC)
  # Copy migration DLLs (both general and voxel-specific)
  file(GLOB MIGRATION_DLLS CONFIGURE_DEPENDS "${MIGRATION_ROOT}/build/RelWithDebInfo/*.dll")
  file(GLOB VOXEL_MIGRATION_DLLS CONFIGURE_DEPENDS "${MIGRATION_ROOT}/build/voxelslam/RelWithDebInfo/*.dll")

  if(MIGRATION_DLLS OR VOXEL_MIGRATION_DLLS)
    add_custom_command(TARGET voxelslam POST_BUILD
      COMMAND ${CMAKE_COMMAND} -E make_directory "$<TARGET_FILE_DIR:voxelslam>"
      COMMENT "Copying migration DLLs to output directory"
    )
    if(MIGRATION_DLLS)
      add_custom_command(TARGET voxelslam POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${MIGRATION_DLLS}
          "$<TARGET_FILE_DIR:voxelslam>"
      )
    endif()
    if(VOXEL_MIGRATION_DLLS)
      add_custom_command(TARGET voxelslam POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
          ${VOXEL_MIGRATION_DLLS}
          "$<TARGET_FILE_DIR:voxelslam>"
      )
    endif()
  endif()
endif()

# ===== Install =====
install(TARGETS voxelslam RUNTIME DESTINATION bin)
