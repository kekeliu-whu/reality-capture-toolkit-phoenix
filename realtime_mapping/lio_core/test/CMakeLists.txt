cmake_minimum_required(VERSION 3.10)
project(test)
SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_CXX_FLAGS_DEBUG "$ENV{CXXFLAGS} -O0 -Wall -g -ggdb")
SET(CMAKE_CXX_FLAGS_RELEASE "$ENV{CXXFLAGS} -O3 -Wall")

include_directories(${PROJECT_SOURCE_DIR}/../src ${PROJECT_SOURCE_DIR}/../include)
# aux_source_directory(${PROJECT_SOURCE_DIR}/../src SOURCE)
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)
message(status "out_put_path:${EXECUTABLE_OUTPUT_PATH}")
find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(undistort undistort_test.cpp ../src/interface/undistort.cpp) #
target_link_libraries(undistort ${PCL_LIBRARIES})