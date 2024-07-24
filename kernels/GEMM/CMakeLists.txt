cmake_minimum_required(VERSION 3.22.0)

project(CUDA_GEMM LANGUAGES CXX CUDA)



set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_89,code=sm_89")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/build)

set(MKL_LINK "static")
set(MKL_INTERFACE_FULL "intel_lp64")

find_package(CUDA REQUIRED)
find_package(OpenMP REQUIRED)
find_package(MKL CONFIG REQUIRED)




include_directories(${PROJECT_SOURCE_DIR}/include)
include_directories(${MKL_INCLUDE_DIRS})


file(GLOB SOURCES ${PROJECT_SOURCE_DIR}/src/*)
list(APPEND SOURCES ${PROJECT_SOURCE_DIR}/main.cc)


add_executable(CUDA_GEMM ${SOURCES})


target_link_libraries(CUDA_GEMM ${CUDA_LIBRARIES})
target_link_libraries(CUDA_GEMM OpenMP::OpenMP_CXX)
target_link_libraries(CUDA_GEMM ${MKL_LIBRARIES})
link_directories(${MKL_LIBRARIES_DIRS})




