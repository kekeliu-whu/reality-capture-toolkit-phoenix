cmake_minimum_required(VERSION 3.10)
cmake_policy(SET CMP0048 NEW)
project(migration_core VERSION 0.0.1)

set(CMAKE_CXX_STANDARD 17)
if (CMAKE_CXX_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_CXX_FLAGS "-Wthread-safety")
endif()

find_package(PCL REQUIRED COMPONENTS common io)
find_package(Ceres REQUIRED)
find_package(Protobuf REQUIRED)
find_package(Boost REQUIRED COMPONENTS iostreams)
if(WIN32)
  find_package(Glog REQUIRED)
endif()
find_package(GTest REQUIRED)

if (Protobuf_VERBOSE)
    message(STATUS "Using Protocol Buffers ${protobuf_VERSION}")
endif ()

add_subdirectory(thirdparty/sophus)
add_subdirectory(thirdparty/ros)


#################### generate proto srcs and headers ####################
file(GLOB_RECURSE ALL_PROTOS "proto/*.proto")

# TODO(cschuet): Move proto compilation to separate function.
set(ALL_PROTO_SRCS)
set(ALL_PROTO_HDRS)
foreach(ABS_FIL ${ALL_PROTOS})
    file(RELATIVE_PATH REL_FIL ${PROJECT_SOURCE_DIR} ${ABS_FIL})
    get_filename_component(DIR ${REL_FIL} DIRECTORY)
    get_filename_component(FIL_WE ${REL_FIL} NAME_WE)

    list(APPEND ALL_PROTO_SRCS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc")
    list(APPEND ALL_PROTO_HDRS "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h")

    add_custom_command(
        OUTPUT "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.cc"
            "${PROJECT_BINARY_DIR}/${DIR}/${FIL_WE}.pb.h"
        COMMAND  ${PROTOBUF_PROTOC_EXECUTABLE}
        ARGS --cpp_out  ${PROJECT_BINARY_DIR} -I
        ${PROJECT_SOURCE_DIR} ${ABS_FIL}
        DEPENDS ${ABS_FIL}
        COMMENT "Running C++ protocol buffer compiler on ${ABS_FIL}"
        VERBATIM
    )
endforeach()
set_source_files_properties(${ALL_PROTO_SRCS} ${ALL_PROTO_HDRS} PROPERTIES GENERATED TRUE)
#################### generate proto srcs and headers ####################

include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
    ${Boost_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS}
)

list(APPEND PROJECT_SRCS ${ALL_PROTO_SRCS})
list(APPEND
    PROJECT_SRCS
    src/sensor_io.cc
    src/utils.cc
)

set(
    TEST_EXECUTABLE_COMMON_DEPS
    protobuf::libprotobuf
    ${Boost_LIBRARIES}
    glog::glog
    sophus
    rosbag
)

# add_executable(test_proto src/test_proto.cc ${PROJECT_SRCS})
# target_link_libraries(test_proto ${CERES_LIBRARIES} ${PCL_LIBRARIES} protobuf::libprotobuf)

function(add_test_library_srcs ARG_SRC)
    add_library(${TEST_LIB} ${ARG_SRC})
    target_link_libraries(${TEST_LIB} ${TEST_EXECUTABLE_COMMON_DEPS})
endfunction()

include(cmake/google-test.cmake)
set(TEST_LIB ${PROJECT_NAME})
add_test_library_srcs("${PROJECT_SRCS}")
enable_automatic_test_and_benchmark()

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)
