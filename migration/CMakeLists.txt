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
find_package(Glog REQUIRED)
find_package(GTest REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(fmt REQUIRED)
find_package(spdlog REQUIRED)
find_package(PDAL REQUIRED)

if (Protobuf_VERBOSE)
    message(STATUS "Using Protocol Buffers ${protobuf_VERSION}")
endif ()

add_subdirectory(thirdparty/sophus)
add_subdirectory(thirdparty/ulog_cpp)
add_subdirectory(thirdparty/ros)

if(MSVC)
  add_compile_options(/MP)
  set(CMAKE_CXX_FLAGS_RELWITHDEBINFO "/O2 /Ob3 /MD /Zi")
  add_compile_options($<$<CONFIG:Release>:/Zi>)
  add_link_options($<$<CONFIG:Release>:/DEBUG>)
endif()


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

set(PROJECT_SRCS
    include/migration/logging.h
    include/migration/proto_io.h
    include/migration/utils.h
    include/migration/inc_las_writer.h
    include/migration/string.h
    src/proto_io.cc
    src/utils.cc
    src/logging.cc
    src/inc_las_writer.cc
    src/string.cc
    ${ALL_PROTO_HDRS}
    ${ALL_PROTO_SRCS}
)

set(
    TEST_EXECUTABLE_COMMON_DEPS
    protobuf::libprotobuf
    ${Boost_LIBRARIES}
    glog::glog
    sophus
    rosbag
    OpenSSL::SSL
    OpenSSL::Crypto
    yaml-cpp::yaml-cpp
    ulog_cpp
    spdlog::spdlog_header_only
    $<TARGET_NAME:pdalcpp>
    fmt::fmt-header-only
)

# add_executable(test_proto src/test_proto.cc ${PROJECT_SRCS})
# target_link_libraries(test_proto ${CERES_LIBRARIES} ${PCL_LIBRARIES} protobuf::libprotobuf)

include(cmake/google-test.cmake)
set(TEST_LIB ${PROJECT_NAME})
add_test_library_srcs("${PROJECT_SRCS}")
enable_automatic_test_and_benchmark()

add_executable(convert_l2pro src/tools/convert_l2pro.cc)
target_link_libraries(convert_l2pro ${TEST_LIB})
add_executable(convert_s20 src/tools/convert_s20.cc)
target_link_libraries(convert_s20 ${TEST_LIB})

target_include_directories(
    ${PROJECT_NAME}
    PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_BINARY_DIR}
)
