cmake_minimum_required(VERSION 3.8)
project(zsph)

include_directories("../header/")

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

file(GLOB_RECURSE files "./*.cu" "../equation/*.cu" "../io/*.cu" "../lib/*.cu" "../rest_case/*.cu")

message(STATUS "cuda version: "${CUDA_VERSION_STRING})

set(CUDA_SEPARABLE_COMPILATION ON)

cuda_add_executable(zsph ${files})
target_link_libraries(zsph ${CUDA_LIBRARIES} ${VTK_LIBRARIES} -lcudadevrt)
vtk_module_autoinit(
  TARGETS zsph
  MODULES ${VTK_LIBRARIES}
)

