cmake_minimum_required(VERSION 3.2)

set(CMAKE_POSITION_INDEPENDENT_CODE True)
#SET(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
#SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

if(MSVC)
    add_definitions(-DUNICODE -DNOMINMAX)
endif(MSVC)
##########################
#OpenCV
find_package(OpenCV 4 REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

############################
#Eigen
find_package(Eigen3 REQUIRED)
message("Eigen :${EIGEN3_INCLUDE_DIRS}")

############################
#Eigen
find_package(PCL REQUIRED COMPONENTS common io)

#find_package(catkin REQUIRED COMPONENTS cpp_common roscpp_serialization roscpp_traits rostime roslz4)
find_package(Boost REQUIRED COMPONENTS date_time filesystem program_options regex)
if (UNIX)
    find_package(BZip2 REQUIRED)
endif (UNIX)

if (MSVC)
    set(BZIP2_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/bzip2/include)
    set(BZIP2_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/bzip2/lib/bzip2.lib)
endif (MSVC)

if(UNIX)
find_library(lz4_LIBRARIES NAMES lz4)
endif(UNIX)

if (MSVC)
    set(lz4_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/lib/lz4/include)
    set(lz4_LIBRARIES ${CMAKE_CURRENT_SOURCE_DIR}/lib/lz4/lib/lz4.lib)
endif (MSVC)


file(GLOB CONSOLE_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/console_bridge/src/*.cpp")
file(GLOB CPP_COMMON_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/cpp_common/src/*.cpp")
file(GLOB ROSCPP_SERIALIZATION_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/roscpp_serialization/src/*.cpp")
file(GLOB ROSTIME_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/roscpp_core/rostime/src/*.cpp")
file(GLOB CV_BRIDGE_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/cv_bridge/*.cpp")

file(GLOB ROSBAG_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/tools/rosbag_storage/src/*.cpp")
file(GLOB ROSLZ4_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/ros_comm/utilities/roslz4/src/[a-z]*.c")
set(WINDOWS_EXPORT_ALL_SYMBOLS ON)
add_library(rosbag STATIC
        ${ROSBAG_SRCS}
        ${ROSTIME_SRCS}
        ${CPP_COMMON_SRCS}
        ${ROSCPP_SERIALIZATION_SRCS}
        ${ROSLZ4_SRCS}
        ${CONSOLE_BRIDGE_SRCS}
        ${CV_BRIDGE_SRCS}
        ${EIGEN3_INCLUDE_DIRS}
        ${PCL_INCLUDE_DIRS})
#target_compile_definitions(rosbag PUBLIC ROS_BUILD_SHARED_LIBS)

# export common export api
include(GenerateExportHeader)
#generate_export_header(xgrids_ulog EXPORT_FILE_NAME
#        include/xgrids_ulog/export.h)
generate_export_header(rosbag
        BASE_NAME CONSOLE_BRIDGE
        EXPORT_MACRO_NAME CONSOLE_BRIDGE_DLLAPI
        EXPORT_FILE_NAME ${CMAKE_CURRENT_SOURCE_DIR}/include/console_bridge_export.h

        )

target_include_directories(rosbag PUBLIC
        include
        pcl_conversions/include
        console_bridge/include
        roscpp_core/cpp_common/include
        roscpp_core/rostime/include
        roscpp_core/roscpp_serialization/include
        roscpp_core/roscpp_traits/include
        ros_comm/utilities/roslz4/include
        ros_comm/tools/rosbag_storage/include)

if(MSVC)
    target_include_directories(rosbag PUBLIC
       ${BZIP2_INCLUDE_DIRS}
        ${lz4_INCLUDE_DIRS}
    )
endif()


target_link_libraries(rosbag PUBLIC
        Boost::date_time
        Boost::filesystem
        Boost::program_options
        Boost::regex
        ${BZIP2_LIBRARIES}
        ${lz4_LIBRARIES}
        ${OpenCV_LIBS}
        )

