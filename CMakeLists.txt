PROJECT( eyelike_project )
FIND_PACKAGE( OpenCV REQUIRED )
cmake_minimum_required(VERSION 2.8)

SET(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin)
set(CMAKE_BINARY_DIR ${PROJECT_BINARY_DIR}/bin)

add_subdirectory(src)