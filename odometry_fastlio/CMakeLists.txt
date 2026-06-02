cmake_minimum_required(VERSION 3.10)
project(odometry_fastlio)

if(CMAKE_TOOLCHAIN_FILE)
    include(${CMAKE_TOOLCHAIN_FILE})
endif(CMAKE_TOOLCHAIN_FILE)

if(MSVC)
  add_compile_options(/MP /bigobj /utf-8)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob3 /MD /Zi")
  add_compile_options($<$<CONFIG:Release>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/DEBUG>)
endif()

add_definitions(-D_USE_MATH_DEFINES)
add_definitions(-DGLOG_USE_GLOG_EXPORT)
add_definitions(-DROOT_DIR="D:/ProjectX/project-3d/reality-capture-toolkit/odometry_fastlio/")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(gflags REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(OpenMP QUIET)

# pthread for ikd-Tree multi-threaded rebuild
find_library(PTHREAD_LIBRARY NAMES pthreadVC3 pthread NAMES_PER_DIR)
if(PTHREAD_LIBRARY)
  message(STATUS "Found pthread: ${PTHREAD_LIBRARY}")
else()
  message(WARNING "pthread not found, ikd-Tree rebuild may fail to link")
endif()

add_subdirectory(../migration migration_build)

# CPU core detection for OpenMP
include(ProcessorCount)
ProcessorCount(N)
message("Processor count: ${N}")
if(N GREATER 4)
  add_definitions(-DMP_EN)
  add_definitions(-DMP_PROC_NUM=3)
  message("OpenMP cores: 3")
elseif(N GREATER 3)
  add_definitions(-DMP_EN)
  add_definitions(-DMP_PROC_NUM=2)
  message("OpenMP cores: 2")
else()
  add_definitions(-DMP_PROC_NUM=1)
endif()

# Include directories
include_directories(
  include
  include/ikd-Tree
  include/IKFoM_toolkit
  include/IKFoM_toolkit/esekfom
  include/IKFoM_toolkit/mtk
  include/IKFoM_toolkit/mtk/types
  src
  ${EIGEN3_INCLUDE_DIRS}
  ${PCL_INCLUDE_DIRS}
)

set(FASTLIO_SRCS
  src/laserMapping.cpp
  src/preprocess.cpp
  include/ikd-Tree/ikd_Tree.cpp
)

set(FASTLIO_LIBS
  ${PCL_LIBRARIES}
  ${GFLAGS_LIBRARIES}
  yaml-cpp::yaml-cpp
  fmt::fmt-header-only
  spdlog::spdlog_header_only
  migration_core
  rosbag
)

if(OpenMP_FOUND)
  set(FASTLIO_LIBS ${FASTLIO_LIBS} OpenMP::OpenMP_CXX)
endif()

if(PTHREAD_LIBRARY)
  set(FASTLIO_LIBS ${FASTLIO_LIBS} ${PTHREAD_LIBRARY})
endif()

# Main offline executable
add_executable(fastlio_offline
  src/offline_main.cpp
  ${FASTLIO_SRCS}
)

target_link_libraries(fastlio_offline ${FASTLIO_LIBS})

# Also expose the mapping sources as a library for potential reuse
add_library(fastlio_mapping STATIC ${FASTLIO_SRCS})
target_link_libraries(fastlio_mapping PUBLIC ${FASTLIO_LIBS})

# Install
install(TARGETS fastlio_offline RUNTIME DESTINATION bin)
