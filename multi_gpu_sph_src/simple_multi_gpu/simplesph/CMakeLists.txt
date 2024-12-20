cmake_minimum_required(VERSION 3.8)
project(simplesph)

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

#set(CUDA_NVCC_FLAGS -g;-G)
set(CUDA_NVCC_FLAGS -lineinfo;--ptxas-options=-v;-arch=compute_89;-code=sm_89;-maxrregcount 64)
file(GLOB_RECURSE files "../io/*.cu" "../lib/*.cu" "../equation/*.cu" "./*.cu" )

add_definitions(-DZSPH_BINARY)
add_definitions(-DZSPH_OUTPUT_TYPE)
add_definitions(-DZSPH_OUTPUT_P)
add_definitions(-DZSPH_OUTPUT_VEL)
add_definitions(-DZSPH_OUTPUT_ACC)
add_definitions(-DZSPH_OUTPUT_HASH)
add_definitions(-DZSPH_OUTPUT_ISPTC)
add_definitions(-DZSPH_OUTPUT_GPUID)
#add_definitions(-DZSPH_OUTPUT_WSUM)

message(STATUS "cuda version: "${CUDA_VERSION_STRING})

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(simplesph ${files})
target_link_libraries(simplesph ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS simplesph
  MODULES ${VTK_LIBRARIES}
)
