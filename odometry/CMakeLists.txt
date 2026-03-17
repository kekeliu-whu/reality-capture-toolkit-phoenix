cmake_minimum_required(VERSION 3.10)
project(odometry)

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
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob3 /MD /Zi")
  add_compile_options($<$<CONFIG:Release>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/DEBUG>)
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
find_package(OpenCV CONFIG REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(RapidJSON CONFIG REQUIRED)
find_package(Glog REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(mimalloc CONFIG REQUIRED)
find_package(crashpad CONFIG REQUIRED)
find_package(PDAL CONFIG REQUIRED)
find_package(OpenMP CONFIG REQUIRED)
find_package(spdlog CONFIG REQUIRED)

include_directories(
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${CERES_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

add_subdirectory(../migration migration_build)


set(
  PGO_PRIVATE_SRCS
  src/core/slam_core_lib.cc
  src/core/preprocess.cc
  src/io/msg_pack_synchronizer.cc
  src/common/msg_conversions.cc
)

set(PGO_PRIVATE_LIBS
  mimalloc
  migration_core
  ${GFLAGS_LIBRARIES}
  ${PCL_LIBRARIES}
  ${CERES_LIBRARIES}
  glog::glog
  absl::flat_hash_map
  absl::flat_hash_set
  rapidjson
  crashpad::crashpad
  $<TARGET_NAME:pdalcpp>
  OpenMP::OpenMP_CXX
  spdlog::spdlog_header_only
)

function(add_private_tool name)
  add_executable(${name} ${ARGN} ${PGO_PRIVATE_SRCS})
  target_link_libraries(${name} ${PGO_PRIVATE_LIBS})
endfunction()

add_private_tool(slam_core_main src/slam_core_main.cc)
add_private_tool(accuracy_evaluation src/tools/accuracy_evaluation.cc)
