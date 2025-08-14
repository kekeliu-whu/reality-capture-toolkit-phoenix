
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

add_library(${PROJECT_NAME}
  STATIC
	data_container.cpp
	messages.cpp
	reader.cpp
	writer.cpp
	simple_writer.cpp
)
add_library(${PROJECT_NAME}::${PROJECT_NAME} ALIAS ${PROJECT_NAME})

