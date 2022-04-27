set(CMAKE_C_COMPILER gcc)
cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PROJECT LANGUAGES CXX CUDA)
find_package(CUDA 10.1 REQUIRED)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)

if (CMAKE_BUILD_TYPE STREQUAL "Debug")
  message("Debug mode")
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;-std=c++11;-g;-lineinfo;-Xcompiler;-ggdb)
  else()
    set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode;-std=c++11;-O3;-DNDEBUG;-Xcompiler;-DNDEBUG)
endif()

include(CTest)
  
add_library(ann STATIC src/ann.cu)

target_include_directories(ann PUBLIC ${CMAKE_CURRENT_LIST_DIR}/include)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")
 
# We need to explicitly state that we need all CUDA files in the 
# library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(ann PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ann PRIVATE ${CUDA_LIBRARIES})

set(MNIST_DATA_DIR ${CMAKE_CURRENT_LIST_DIR}/data)
target_compile_definitions(ann PRIVATE MNIST_DATA_DIR="${MNIST_DATA_DIR}")

add_executable(ann_test tests/test.cu)
 
set_property(TARGET ann_test PROPERTY CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(ann_test PRIVATE ann)
 
if(APPLE)
  # We need to add the path to the driver (libcuda.dylib) as an rpath, 
  # so that the static cuda runtime can find it at runtime.
  set_property(TARGET ann_test 
               PROPERTY
               BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()