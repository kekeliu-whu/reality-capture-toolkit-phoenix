cmake_minimum_required(VERSION 3.16)
project(navvis_cleanroom_reconstruction LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 3.3 REQUIRED NO_MODULE)
find_package(OpenCV 4 REQUIRED COMPONENTS core imgproc photo imgcodecs stitching)
find_package(OpenMP)
find_package(Ceres QUIET)
find_package(PkgConfig REQUIRED)
pkg_check_modules(ROS1 REQUIRED IMPORTED_TARGET rosbag sensor_msgs)
find_library(
    BrotliEncoder_LIBRARY
    NAMES brotlienc libbrotlienc.so.1
    REQUIRED)
find_path(
    PCL_HEADER_ONLY_INCLUDE_DIR
    NAMES pcl/octree/octree_search.h
    PATH_SUFFIXES pcl-1.12 pcl-1.13 pcl-1.14)
if(NOT PCL_HEADER_ONLY_INCLUDE_DIR)
    message(FATAL_ERROR
        "PCL octree headers are required for the byte-exact panorama sparse-depth renderer. "
        "Install libpcl-dev or pass -DPCL_HEADER_ONLY_INCLUDE_DIR=/path/to/pcl/include.")
endif()
find_path(
    LibRaw_INCLUDE_DIR
    NAMES libraw/libraw.h
    HINTS "${CMAKE_CURRENT_SOURCE_DIR}/third_party/libraw/include")
# nv_image-postprocessing is linked to LibRaw 0.22 (SONAME liblibraw.so.24).
# Its public C structure is ABI-versioned, so compiling against the older
# Debian 0.20 headers while loading .24 silently writes the wrong offsets.
find_file(
    LibRaw22_LIBRARY
    NAMES liblibraw.so.24
    PATHS /lib /usr/lib /usr/local/lib
    PATH_SUFFIXES x86_64-linux-gnu aarch64-linux-gnu)

# Small C ABI used by the Python SLAM reconstruction.  The installed matcher
# stores surfel statistics as float and updates them one hit at a time; doing
# that loop in Python would dominate every frontend regression.
add_library(
    navvis_recon_slam_frontend_native SHARED
    src/slam_frontend_native.cpp)
target_link_libraries(navvis_recon_slam_frontend_native PRIVATE Eigen3::Eigen)
target_compile_options(
    navvis_recon_slam_frontend_native PRIVATE
    -Wall -Wextra -Wpedantic -ffp-contract=off)
if(OpenMP_CXX_FOUND)
    target_link_libraries(
        navvis_recon_slam_frontend_native PRIVATE OpenMP::OpenMP_CXX)
endif()

# Complete C++ SurveyorSLAM frontend.  Python remains an offline oracle only;
# the production executable reads NVSLAM6 and ROS1 IMU input directly.
add_library(
    navvis_recon_slam_runtime
    src/slam_archive.cpp
    src/slam_imu.cpp
    src/slam_batch_collator.cpp
    src/slam_frontend.cpp
    src/slam_rosbag.cpp)
target_include_directories(navvis_recon_slam_runtime PUBLIC include)
target_link_libraries(
    navvis_recon_slam_runtime
    PUBLIC Eigen3::Eigen PkgConfig::ROS1
    PRIVATE navvis_recon_slam_frontend_native)
target_compile_options(
    navvis_recon_slam_runtime PRIVATE
    -Wall -Wextra -Wpedantic -ffp-contract=off)

add_executable(navvis_recon_slam apps/slam_pipeline.cpp)
target_link_libraries(navvis_recon_slam PRIVATE navvis_recon_slam_runtime)

add_executable(navvis_recon_slam_archive_test tests/slam_archive_test.cpp)
target_link_libraries(navvis_recon_slam_archive_test PRIVATE navvis_recon_slam_runtime)

add_executable(navvis_recon_slam_imu_test tests/slam_imu_test.cpp)
target_link_libraries(navvis_recon_slam_imu_test PRIVATE navvis_recon_slam_runtime)

add_executable(
    navvis_recon_slam_batch_collator_test
    tests/slam_batch_collator_test.cpp)
target_link_libraries(
    navvis_recon_slam_batch_collator_test PRIVATE navvis_recon_slam_runtime)

add_executable(navvis_recon_slam_rosbag_test tests/slam_rosbag_test.cpp)
target_link_libraries(navvis_recon_slam_rosbag_test PRIVATE navvis_recon_slam_runtime)

add_library(navvis_recon
    src/binary_surface_pipeline.cpp
    src/cloud_builder.cpp
    src/cloud_surface_filter.cpp
    src/mapped_space_quality.cpp
    src/pointcloud_coloring.cpp
    src/image_postprocessing.cpp
    src/panorama_rendering.cpp)
target_include_directories(navvis_recon PUBLIC include)
target_include_directories(navvis_recon SYSTEM PRIVATE ${PCL_HEADER_ONLY_INCLUDE_DIR})
target_link_libraries(navvis_recon PUBLIC Eigen3::Eigen ${OpenCV_LIBS} ${BrotliEncoder_LIBRARY})
target_compile_options(navvis_recon PRIVATE -Wall -Wextra -Wpedantic -ffp-contract=off)
if(OpenMP_CXX_FOUND)
    target_link_libraries(navvis_recon PUBLIC OpenMP::OpenMP_CXX)
endif()

add_executable(navvis_recon_smoke tests/smoke.cpp)
target_link_libraries(navvis_recon_smoke PRIVATE navvis_recon)

add_executable(navvis_recon_panorama_sampling tests/panorama_sampling.cpp)
target_link_libraries(navvis_recon_panorama_sampling PRIVATE navvis_recon)

add_executable(navvis_recon_mapped_space_quality_test tests/mapped_space_quality.cpp)
target_link_libraries(navvis_recon_mapped_space_quality_test PRIVATE navvis_recon)

add_executable(
    navvis_recon_surface_capture_acceptance
    tests/surface_capture_acceptance.cpp)
target_link_libraries(navvis_recon_surface_capture_acceptance PRIVATE navvis_recon)

add_executable(navvis_recon_pandar apps/pandar_cloud_pipeline.cpp)
target_link_libraries(navvis_recon_pandar PRIVATE navvis_recon)

add_executable(navvis_recon_shard_surface_filter apps/pandar_shard_surface_filter.cpp)
target_link_libraries(navvis_recon_shard_surface_filter PRIVATE navvis_recon)

add_executable(navvis_recon_mapped_space_quality apps/mapped_space_quality.cpp)
target_link_libraries(navvis_recon_mapped_space_quality PRIVATE navvis_recon)

add_executable(navvis_recon_surface_colorizer apps/surface_panorama_colorizer.cpp)
target_link_libraries(navvis_recon_surface_colorizer PRIVATE navvis_recon)
if(Ceres_FOUND)
    if(DEFINED Ceres_VERSION)
        message(STATUS "Using Ceres ${Ceres_VERSION} for exposure alignment")
        if(Ceres_VERSION VERSION_LESS 2.2)
            message(WARNING
                "Ceres ${Ceres_VERSION} solves the same exposure objective, but the "
                "validated 32-thread convergence distribution is reproduced with "
                "Ceres 2.2.x. Use Ceres >= 2.2 for standard color regressions.")
        endif()
    endif()
    target_link_libraries(navvis_recon_surface_colorizer PRIVATE Ceres::ceres)
    # Ceres is intentionally loaded from the bundled Ubuntu 22.04 runtime.
    # Emit DT_RPATH instead of DT_RUNPATH so its indirect SuiteSparse
    # dependencies (for example libspqr.so.2) are resolved from that same
    # self-contained runtime without requiring LD_PRELOAD/LD_LIBRARY_PATH.
    target_link_options(
        navvis_recon_surface_colorizer PRIVATE LINKER:--disable-new-dtags)
    target_compile_definitions(navvis_recon_surface_colorizer PRIVATE NAVVIS_RECON_HAVE_CERES=1)

    add_executable(
        navvis_recon_stage1_imu_ceres_solver
        apps/stage1_imu_ceres_solver.cpp)
    target_link_libraries(
        navvis_recon_stage1_imu_ceres_solver PRIVATE Ceres::ceres Eigen3::Eigen)
    target_link_options(
        navvis_recon_stage1_imu_ceres_solver PRIVATE LINKER:--disable-new-dtags)
    target_compile_options(
        navvis_recon_stage1_imu_ceres_solver PRIVATE -Wall -Wextra -Wpedantic)

    add_executable(
        navvis_recon_stage2_imu_ceres_solver
        apps/stage2_imu_ceres_solver.cpp)
    target_link_libraries(
        navvis_recon_stage2_imu_ceres_solver PRIVATE Ceres::ceres Eigen3::Eigen)
    target_link_options(
        navvis_recon_stage2_imu_ceres_solver PRIVATE LINKER:--disable-new-dtags)
    target_compile_options(
        navvis_recon_stage2_imu_ceres_solver PRIVATE -Wall -Wextra -Wpedantic)
else()
    message(WARNING
        "Ceres not found; the surface colorizer will use the slower finite-difference "
        "exposure solver and the native Stage1/Stage2 IMU solvers will not be built")
endif()

if(LibRaw_INCLUDE_DIR AND LibRaw22_LIBRARY)
    add_executable(navvis_recon_ocam_panorama apps/ocam_panorama_pipeline.cpp)
    target_include_directories(navvis_recon_ocam_panorama PRIVATE ${LibRaw_INCLUDE_DIR})
    target_link_libraries(
        navvis_recon_ocam_panorama PRIVATE navvis_recon ${LibRaw22_LIBRARY} ${CMAKE_DL_LIBS})
    target_compile_options(navvis_recon_ocam_panorama PRIVATE -Wall -Wextra -Wpedantic)
else()
    message(WARNING "LibRaw 0.22/liblibraw.so.24 not found; navvis_recon_ocam_panorama will not be built")
endif()

enable_testing()
add_test(NAME navvis_recon_smoke COMMAND navvis_recon_smoke)
add_test(NAME navvis_recon_panorama_sampling COMMAND navvis_recon_panorama_sampling)
add_test(NAME navvis_recon_mapped_space_quality COMMAND navvis_recon_mapped_space_quality_test)
add_test(NAME navvis_recon_slam_imu COMMAND navvis_recon_slam_imu_test)
