project(log)
set(CMAKE_CXX_STANDARD 14)
find_package(spdlog CONFIG REQUIRED)
get_target_property(SPDLOG_INCLUDE_DIRS spdlog::spdlog INTERFACE_INCLUDE_DIRECTORIES)

set(CURRENT_SOURCES ${CMAKE_CURRENT_SOURCE_DIR}/src/lsLogger.cpp)

add_library(${PROJECT_NAME} STATIC ${CURRENT_SOURCES})
add_library(xgrids::log ALIAS ${PROJECT_NAME})

target_include_directories(${PROJECT_NAME} PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
#target_compile_definitions(${PROJECT_NAME} PRIVATE -DXGRIDS_TOOL_DLLEXPORT)
target_link_libraries(${PROJECT_NAME} PUBLIC
        spdlog::spdlog
        )
target_compile_definitions(${PROJECT_NAME} PRIVATE -DPluginName)
