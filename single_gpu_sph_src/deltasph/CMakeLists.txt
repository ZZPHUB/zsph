cmake_minimum_required(VERSION 3.8)
project(deltasph)

include_directories("../header/" "../../thirdparty/json/include")

find_package(CUDA REQUIRED)
find_package(VTK COMPONENTS
  ChartsCore
  CommonColor
  CommonCore
  CommonDataModel
  FiltersCore
  FiltersGeneral
  FiltersSources
  IOLegacy
  InteractionStyle
)
#include(${VTK_USE_FILE})
#include(${CUDA_INCLUDE_DIRS})

#set(CUDA_NVCC_FLAGS -g;-G)
set(CUDA_NVCC_FLAGS -lineinfo;--ptxas-options=-v;-arch=compute_75;-code=sm_75;-maxrregcount 64)
file(GLOB_RECURSE files "./*.cu" "../equation/delta_ns.cu" "../io/*.cu" "../lib/*.cu")

add_definitions(-DZSPH_BINARY)
add_definitions(-DZSPH_OUTPUT_TYPE)
add_definitions(-DZSPH_OUTPUT_P)
add_definitions(-DZSPH_OUTPUT_VEL)
add_definitions(-DZSPH_OUTPUT_ACC)
add_definitions(-DZSPH_DELTA)
#add_definitions(-DZSPH_OUTPUT_HASH)


message(STATUS "cuda version: "${CUDA_VERSION_STRING})

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(deltasph ${files})
target_link_libraries(deltasph ${CUDA_LIBRARIES} ${VTK_LIBRARIES} -lcudadevrt)
vtk_module_autoinit(
  TARGETS deltasph
  MODULES ${VTK_LIBRARIES}
)
