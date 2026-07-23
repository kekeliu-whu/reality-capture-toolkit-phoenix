cmake_minimum_required(VERSION 2.8.3)
project(xmap)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)
if(MSVC)
    # The sources are UTF-8 and contain Chinese comments.  Without /utf-8,
    # MSVC interprets them as CP936 and can accidentally continue a comment
    # onto the following type declaration.
    add_compile_options(/utf-8 /bigobj)
    add_compile_definitions(NOMINMAX)
else()
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fexceptions")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pthread -fexceptions -DNOMINMAX")
endif()

set(XMAP_ROOT_DIR ${CMAKE_CURRENT_SOURCE_DIR})
set(XMAP_BUILD_DIR ${CMAKE_CURRENT_BINARY_DIR})
set(XMAP_CONFIG_PATH "${XMAP_ROOT_DIR}/configs")
set(XMAP_TEST_DATA_PATH "${XMAP_ROOT_DIR}/unit_test/test_data/")

# map.yaml is optional for the offline app, which passes its complete runtime
# configuration explicitly. Some source packages do not ship the example
# configs directory.
if(EXISTS "${XMAP_CONFIG_PATH}/map.yaml")
    file(COPY ${XMAP_CONFIG_PATH}/map.yaml DESTINATION ${CMAKE_CURRENT_BINARY_DIR})
endif()

SET(CMAKE_BUILD_TYPE "Release")
# SET(CMAKE_BUILD_TYPE "RelWithDebInfo")
if (${CMAKE_BUILD_TYPE} MATCHES "Debug")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
elseif (${CMAKE_BUILD_TYPE} MATCHES "RelWithDebInfo")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -g")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g")
    set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -g")
    SET(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -g")
else ()
    SET(CMAKE_BUILD_TYPE "Release")
    if(NOT MSVC)
        set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")
    endif()
endif ()


SET(XMAP_TEST_OPTION OFF)

find_package(Eigen3 REQUIRED)
find_package(PCL REQUIRED)
find_package(yaml-cpp REQUIRED)

add_library(xmap_lib STATIC
        src/xmap.cpp
        src/xmap_util.cpp
        src/xmap_dynamic.cpp
        src/xmap_maintain.cpp
        src/xmap_search.cpp
        src/voxel_loc.cpp
        src/small_voxel_value.cpp

        # # 合并之后可重构去除
        # third_party/mem_monitor.cpp
        )
target_include_directories(xmap_lib PUBLIC ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS}
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party
        ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(xmap_lib ${PCL_LIBRARIES} ${YAML_CPP_LIBRARIES})

if (XMAP_TEST_OPTION)
    message("XMAP_TEST_OPTION is enabled")
    find_package(libLAS REQUIRED)
    find_package(GTest REQUIRED)

    set(XMAP_TEST_FILES
            unit_test/plane_fitting_test.cpp
            unit_test/incremental_test.cpp        # 涉及xivt与xmap的对比测试，建议往上一级构建
            unit_test/normal_filter_test.cpp
            unit_test/search_performace_test.cpp  # 涉及xivt与xmap的对比测试，建议往上一级构建
            unit_test/hash_test.cpp
            unit_test/search_acc_test.cpp         # 涉及xivt与xmap的对比测试，建议往上一级构建
            unit_test/octo_search_test.cpp
            unit_test/unorderedmap_test.cpp
            unit_test/forget_test.cpp
            unit_test/dynamic_save_load_test.cpp
            )

    foreach (TEST_FILE ${XMAP_TEST_FILES})
        get_filename_component(TEST_NAME ${TEST_FILE} NAME_WE)
        add_executable(${TEST_NAME} ${TEST_FILE})
        target_include_directories(${TEST_NAME} PRIVATE ${EIGEN3_INCLUDE_DIR} ${PCL_INCLUDE_DIRS} ${YAML_CPP_INCLUDE_DIRS}
                ${CMAKE_CURRENT_SOURCE_DIR}/third_party
                ${CMAKE_CURRENT_SOURCE_DIR}/include)
        target_link_libraries(${TEST_NAME} PRIVATE xmap_lib GTest::GTest GTest::Main ${libLAS_LIBRARIES})
    endforeach ()
endif ()
