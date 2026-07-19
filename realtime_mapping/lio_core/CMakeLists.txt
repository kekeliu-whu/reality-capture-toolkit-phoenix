cmake_minimum_required(VERSION 3.10)
project(lio_core)

SET(CMAKE_BUILD_TYPE "Release")
add_definitions(-DROOT_DIR=\"${CMAKE_CURRENT_SOURCE_DIR}/\")

set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -std=c++17 -pthread  -fexceptions")

set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake ${CMAKE_MODULE_PATH})
# include(version.cmake)

include(ProcessorCount)
ProcessorCount(NPROC)

if (NPROC GREATER 6)
    set(NPROC 6)
endif ()

MATH(EXPR NPROC "${NPROC}-2")
message("core for MP: ${NPROC}")

find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}   ${OpenMP_C_FLAGS}")
message("Current CPU archtecture: ${CMAKE_SYSTEM_PROCESSOR}")
if (CMAKE_SYSTEM_PROCESSOR MATCHES "(x86)|(X86)|(amd64)|(AMD64)")
    include(ProcessorCount)
    ProcessorCount(N)
    message("Processer number:  ${N}")
    if (N GREATER 5)
        math(EXPR PROC_NUM "${N} - 2")
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM=${PROC_NUM})
        message("core for MP:  ${PROC_NUM}")
    elseif (N GREATER 3)
        math(EXPR PROC_NUM "${N} - 1")
        add_definitions(-DMP_EN)
        add_definitions(-DMP_PROC_NUM=${PROC_NUM})
        message("core for MP:  ${PROC_NUM}")
    else ()
        add_definitions(-DMP_PROC_NUM=1)
    endif ()
else ()
    add_definitions(-DMP_PROC_NUM=1)
endif ()


set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -fPIC")
SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -fPIC")

if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
endif ()

if (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
endif ()

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(OpenCV 3 REQUIRED)
# find_package(libLAS REQUIRED)
find_package(Gflags)
# find_package(Glog)
find_package(yaml-cpp REQUIRED)
message("PCL libs: ${PCL_LIBRARIES}")
message(Eigen: ${EIGEN3_INCLUDE_DIR})

add_subdirectory(thirdparty/GeographicLib-2.1.1/)
add_subdirectory(thirdparty/xmap/)
add_subdirectory(thirdparty/lslog/)

include_directories(
        ${EIGEN3_INCLUDE_DIR}
        ${PCL_INCLUDE_DIRS}
        ${OpenCV_INCLUDE_DIRS}
        ${GFLAGS_INCLUDE_DIRS}
        ${GLOG_INCLUDE_DIRS}
        ${YAML_CPP_INCLUDE_DIRS}
        # ${libLAS_INCLUDE_DIRS}
        interface
        src
        thirdparty/GeographicLib/include
        thirdparty/sophus
        thirdparty/lslog/include
        ${CMAKE_CURRENT_SOURCE_DIR})

set(EXT_LIBS
        ${PCL_LIBRARIES}
        ${OpenCV_LIBRARIES}
        ${GFLAGS_LIBRARIES}
        ${GLOG_LIBRARIES}
        ${YAML_CPP_LIBRARIES}
        # ${libLAS_LIBRARIES}
        GeographicLib
        glog
        xmap_lib
        log
        #   lxutil
        )

add_library(lio_core SHARED
        src/ieskf/ieskf.cpp
        src/ieskf/ieskf_predict.cpp
        src/ieskf/ieskf_update.cpp
        src/ieskf/state_group.cpp
        src/sensor_fusion/lidar_fusion.cpp
        src/sensor_fusion/static_fusion.cpp
        src/sensor_fusion/base_fusion.cpp
        src/lio.cpp
        src/common/common_lib.cpp
        src/method/lio_init.cpp
        src/method/preprocess.cpp
        src/method/io_utils.cpp
        src/method/uniform_sampling.cpp
        src/method/undistort.cpp
        )

target_link_libraries(lio_core ${EXT_LIBS})

## 找到catkin宏和库
find_package(catkin REQUIRED COMPONENTS
        roscpp
        rosbag
        sensor_msgs
        )

## 包含catkin头文件和库
include_directories(
        ${catkin_INCLUDE_DIRS}
)

## 声明catkin包
# catkin_package()

SET(TEST_OPTION OFF)
if (TEST_OPTION)
    message("TEST_OPTION is enabled")
    find_package(libLAS REQUIRED)
    find_package(GTest REQUIRED)

    set(TEST_FILES
            # test/ieskf_static_test.cpp
            # test/ieskf_wall_test.cpp
            # test/rotation_test.cpp
            test/uniform_sampling_test.cpp
            # test/undistort_test.cpp
            )

    foreach (TEST_FILE ${TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_FILE})
        target_include_directories(${TEST_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS}
                ${catkin_LIBRARIES}
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party
                ${CMAKE_CURRENT_SOURCE_DIR}/include/ieskf)
        target_link_libraries(${TEST_NAME} PRIVATE ${catkin_LIBRARIES} ${PCL_LIBRARIES} lio_core GTest::GTest GTest::Main ${libLAS_LIBRARIES})
    endforeach ()
endif ()

# add_library(voxelmap
#   src/voxelmap/octo_tree.cpp
#   src/voxelmap/voxel_map.cpp)
# target_link_libraries(voxelmap  ${EXT_LIBS})

# # Mark executables and/or libraries for installation
# install(TARGETS  lio_core
#   ARCHIVE DESTINATION
#   LIBRARY DESTINATION
#   RUNTIME DESTINATION
# )

# install(DIRECTORY
#   config
#   DESTINATION  ${CMAKE_INSTALL_PREFIX}
# 	)

# install(FILES
#   version.cmake
#   readme.md
#   DESTINATION ${CMAKE_INSTALL_PREFIX}
#   )