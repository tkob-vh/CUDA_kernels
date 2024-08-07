cmake_minimum_required(VERSION 3.22.0)

project(stencil_cuda LANGUAGES CXX CUDA)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_BINARY_DIR})

find_package(CUDA REQUIRED)

include_directories(${CUDA_INCLUDE_DIRS})
include_directories(${PROJECT_SOURCE_DIR}/include)

file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*)
list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/main.cc)

add_executable(stencil ${SOURCES})

target_link_libraries(stencil ${CUDA_LIBRARIES})
