# Auto Generated code - Do not edit.

cmake_minimum_required(VERSION 3.18)
project(CUTLASS_MULTI_GEMMS LANGUAGES CXX CUDA)
find_package(CUDAToolkit)
set(CUDA_PATH ${CUDA_TOOLKIT_ROOT_DIR})
set(CUTLASS_PATH "/scratch/adalal542/KernelBench/sources/cutlass/include")
set(CUTLASS_UTIL_PATH "/scratch/adalal542/KernelBench/sources/cutlass/tools/util/include")
list(APPEND CMAKE_MODULE_PATH ${CUDAToolkit_LIBRARY_DIR})
set(GPU_ARCHS "" CACHE STRING
  "List of GPU architectures (semicolon-separated) to be compiled for.")

if("${GPU_ARCHS}" STREQUAL "")
	set(GPU_ARCHS "70")
endif()

foreach(arch ${GPU_ARCHS})
  set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -gencode arch=compute_${arch},code=sm_${arch}")
	if(SM STREQUAL 70 OR SM STREQUAL 75)
    set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}    -DWMMA")
    set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}  -DWMMA")
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -DWMMA")
	endif()
endforeach()

set(CMAKE_C_FLAGS    "${CMAKE_C_FLAGS}")
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS}")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS}  -Xcompiler -Wall")

set(CMAKE_C_FLAGS_DEBUG    "${CMAKE_C_FLAGS_DEBUG}    -Wall -O0")
set(CMAKE_CXX_FLAGS_DEBUG  "${CMAKE_CXX_FLAGS_DEBUG}  -Wall -O0")
set(CMAKE_CUDA_FLAGS_DEBUG "${CMAKE_CUDA_FLAGS_DEBUG} -O0 -G -Xcompiler -Wall")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-extended-lambda")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr")

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -g -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler=-fno-strict-aliasing")

set(COMMON_HEADER_DIRS
  ${PROJECT_SOURCE_DIR}
  ${CUDAToolkit_INCLUDE_DIRS}
)

set(COMMON_LIB_DIRS
  ${CUDAToolkit_LIBRARY_DIR}
)
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_PATH})
list(APPEND COMMON_HEADER_DIRS ${CUTLASS_UTIL_PATH})
include_directories(
  ${COMMON_HEADER_DIRS}
)

link_directories(
  ${COMMON_LIB_DIRS}
)

add_definitions(-D_GLIBCXX_USE_CXX11_ABI=0)
add_definitions(-DGOOGLE_CUDA=1)

add_executable(sample
  sample/sample.cu
  one_api.cu
)
target_link_libraries(sample PRIVATE
  -lcudart
  ${CMAKE_THREAD_LIBS_INIT}
)

if(NOT DEFINED LIB_INSTALL_PATH)
	set(LIB_INSTALL_PATH ${CMAKE_CURRENT_BINARY_DIR})
endif()
