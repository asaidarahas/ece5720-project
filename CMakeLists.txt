cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(PROJECT LANGUAGES CXX CUDA)

# set(CMAKE_CUDA_COMPILER "/usr/local/cuda/bin/nvcc")
set(CMAKE_CUDA_STANDARD 14)
 
include(CTest)
  
add_library(ann STATIC src/ann.cu)
 
# Request that particles be built with -std=c++11
# As this is a public compile feature anything that links to 
# particles will also build with -std=c++11
# target_compile_features(ann PUBLIC cxx_std_11)
 
# We need to explicitly state that we need all CUDA files in the 
# library to be built with -dc as the member functions 
# could be called by other libraries and executables
set_target_properties(ann PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
 
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