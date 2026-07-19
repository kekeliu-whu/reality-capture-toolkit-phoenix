cmake_minimum_required(VERSION 3.0.2)
project(lio_core_ros)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -Werror=return-type")
SET(CMAKE_BUILD_TYPE "RelWithDebInfo")

find_package(catkin REQUIRED COMPONENTS
        roscpp
        std_msgs
        rosbag
        tf
        )
find_package(PCL REQUIRED)

add_subdirectory(../lio_core lio_core_build)
add_subdirectory(thirdparty/ulog_cpp-591e5f9)

## System dependencies are found with CMake's conventions
# find_package(Boost REQUIRED COMPONENTS system)


## Uncomment this if the package has a setup.py. This macro ensures
## modules and global scripts declared therein get installed
## See http://ros.org/doc/api/catkin/html/user_guide/setup_dot_py.html
# catkin_python_setup()

################################################
## Declare ROS messages, services and actions ##
################################################

## To declare and build messages, services or actions from within this
## package, follow these steps:
## * Let MSG_DEP_SET be the set of packages whose message types you use in
##   your messages/services/actions (e.g. std_msgs, actionlib_msgs, ...).
## * In the file package.xml:
##   * add a build_depend tag for "message_generation"
##   * add a build_depend and a exec_depend tag for each package in MSG_DEP_SET
##   * If MSG_DEP_SET isn't empty the following dependency has been pulled in
##     but can be declared for certainty nonetheless:
##     * add a exec_depend tag for "message_runtime"
## * In this file (CMakeLists.txt):
##   * add "message_generation" and every package in MSG_DEP_SET to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * add "message_runtime" and every package in MSG_DEP_SET to
##     catkin_package(CATKIN_DEPENDS ...)
##   * uncomment the add_*_files sections below as needed
##     and list every .msg/.srv/.action file to be processed
##   * uncomment the generate_messages entry below
##   * add every package in MSG_DEP_SET to generate_messages(DEPENDENCIES ...)

## Generate messages in the 'msg' folder
# add_message_files(
#   FILES
#   Message1.msg
#   Message2.msg
# )

## Generate services in the 'srv' folder
# add_service_files(
#   FILES
#   Service1.srv
#   Service2.srv
# )

## Generate actions in the 'action' folder
# add_action_files(
#   FILES
#   Action1.action
#   Action2.action
# )

## Generate added messages and services with any dependencies listed here
# generate_messages(
#   DEPENDENCIES
#   std_msgs
# )

################################################
## Declare ROS dynamic reconfigure parameters ##
################################################

## To declare and build dynamic reconfigure parameters within this
## package, follow these steps:
## * In the file package.xml:
##   * add a build_depend and a exec_depend tag for "dynamic_reconfigure"
## * In this file (CMakeLists.txt):
##   * add "dynamic_reconfigure" to
##     find_package(catkin REQUIRED COMPONENTS ...)
##   * uncomment the "generate_dynamic_reconfigure_options" section below
##     and list every .cfg file to be processed

## Generate dynamic reconfigure parameters in the 'cfg' folder
# generate_dynamic_reconfigure_options(
#   cfg/DynReconf1.cfg
#   cfg/DynReconf2.cfg
# )

###################################
## catkin specific configuration ##
###################################
## The catkin_package macro generates cmake config files for your package
## Declare things to be passed to dependent projects
## INCLUDE_DIRS: uncomment this if your package contains header files
## LIBRARIES: libraries you create in this project that dependent projects also need
## CATKIN_DEPENDS: catkin_packages dependent projects also need
## DEPENDS: system dependencies of this project that dependent projects also need
catkin_package(
        #  INCLUDE_DIRS include
        #  LIBRARIES lio_core_ros
        #  CATKIN_DEPENDS roscpp std_msgs
        DEPENDS EIGEN3
)

###########
## Build ##
###########

## Specify additional locations of header files
## Your package locations should be listed before other locations
include_directories(
        include
        ${catkin_INCLUDE_DIRS}
        ../lio_core/src/
        ../lio_core/interface/
        ${PCL_INCLUDE_DIRS}
        thirdparty/ulog_cpp-591e5f9
        thirdparty/lixel_msgs-f825749
        ../lio_core/thirdparty/lslog/include
)

## Declare a C++ library
# add_library(${PROJECT_NAME}
#   src/${PROJECT_NAME}/lio_core_ros.cpp
# )

## Add cmake target dependencies of the library
## as an example, code may need to be generated before libraries
## either from message generation or dynamic reconfigure
# add_dependencies(${PROJECT_NAME} ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Declare a C++ executable
## With catkin_make all packages are built within a single CMake context
## The recommended prefix ensures that target names across packages don't collide
add_executable(lio_core_ros_node
        src/lio_core_ros_node.cpp
        src/subscriber.cpp
        src/publisher.cpp
        src/parameters_reader.cpp
        src/ulog.cpp
        src/data_io.cpp
        )

add_library(lio_core_lio_interface SHARED
        src/lio_interface.cpp
        src/subscriber.cpp
        src/publisher.cpp
        src/parameters_reader.cpp
        src/ulog.cpp
        src/data_io.cpp
        )

add_executable(lio_core_lio_interface_demo
        src/lio_interface_demo.cpp
        )

## Rename C++ executable without prefix
## The above recommended prefix causes long target names, the following renames the
## target back to the shorter version for ease of user use
## e.g. "rosrun someones_pkg node" instead of "rosrun someones_pkg someones_pkg_node"
# set_target_properties(${PROJECT_NAME}_node PROPERTIES OUTPUT_NAME node PREFIX "")

## Add cmake target dependencies of the executable
## same as for the library above
# add_dependencies(${PROJECT_NAME}_node ${${PROJECT_NAME}_EXPORTED_TARGETS} ${catkin_EXPORTED_TARGETS})

## Specify libraries to link a library or executable target against
set(TARGET_LIBRARIES
        ${catkin_LIBRARIES}
        lio_core
        glog
        ulog_cpp
        )

# if dont need to save .las, please set SAVE_LAS ON
SET(SAVE_LAS ON)
if (SAVE_LAS)
    message("SAVE_LAS ON!!!!")
    find_package(libLAS REQUIRED)
    add_definitions("-DSAVE_LAS")
    include_directories(${libLAS_INCLUDE_DIRS})
    list(APPEND TARGET_LIBRARIES ${libLAS_LIBRARIES})
endif ()

target_link_libraries(lio_core_ros_node
        ${TARGET_LIBRARIES}
        )

target_link_libraries(lio_core_lio_interface
        ${TARGET_LIBRARIES}
        )

target_link_libraries(lio_core_lio_interface_demo
        lio_core_lio_interface
        )

SET(UTILS_OPTION OFF)
if (UTILS_OPTION)
    message("UTILS_OPTION is enabled")
    find_package(libLAS REQUIRED)
    find_package(GTest REQUIRED)

    set(UTILS_FILES
            utils/transform_xbc_poses.cpp
            utils/evaluate_traj.cpp
            utils/evaluate_diff_xyz.cpp
            utils/minus_rtkline.cpp
            )

    foreach (UTIL_FILE ${UTILS_FILES})
        get_filename_component(TEST_NAME ${UTIL_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${UTIL_FILE})
        target_include_directories(${TEST_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS}
                ${catkin_LIBRARIES}
                ${libLAS_INCLUDE_DIRS}
                thirdparty/lixel_msgs-f825749)
        target_link_libraries(${TEST_NAME} PRIVATE ${catkin_LIBRARIES} ${PCL_LIBRARIES} ${libLAS_LIBRARIES})
    endforeach ()
endif ()

#############
## Install ##
#############

# all install targets should use catkin DESTINATION variables
# See http://ros.org/doc/api/catkin/html/adv_user_guide/variables.html

## Mark executable scripts (Python etc.) for installation
## in contrast to setup.py, you can choose the destination
# catkin_install_python(PROGRAMS
#   scripts/my_python_script
#   DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
# )

## Mark executables for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_executables.html
install(TARGETS ${PROJECT_NAME}_node
        RUNTIME DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
        )

## Mark libraries for installation
## See http://docs.ros.org/melodic/api/catkin/html/howto/format1/building_libraries.html
# install(TARGETS ${PROJECT_NAME}
#   ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
#   RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
# )
install(TARGETS lio_core xmap_lib
        ARCHIVE DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        LIBRARY DESTINATION ${CATKIN_PACKAGE_LIB_DESTINATION}
        RUNTIME DESTINATION ${CATKIN_GLOBAL_BIN_DESTINATION}
        )

## Mark cpp header files for installation
# install(DIRECTORY include/${PROJECT_NAME}/
#   DESTINATION ${CATKIN_PACKAGE_INCLUDE_DESTINATION}
#   FILES_MATCHING PATTERN "*.h"
#   PATTERN ".svn" EXCLUDE
# )

install(DIRECTORY config launch
        DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
        )

## Mark other files for installation (e.g. launch and bag files, etc.)
# install(FILES
#   # myfile1
#   # myfile2
#   DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}
# )

#############
## Testing ##
#############

## Add gtest based cpp test target and link libraries
# catkin_add_gtest(${PROJECT_NAME}-test test/test_lio_core_ros.cpp)
# if(TARGET ${PROJECT_NAME}-test)
#   target_link_libraries(${PROJECT_NAME}-test ${PROJECT_NAME})
# endif()

## Add folders to be run by python nosetests
# catkin_add_nosetests(test)
