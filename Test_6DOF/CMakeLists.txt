cmake_minimum_required(VERSION 3.8)
project(dof)

include_directories("./header/" "../thirdparty/json/include")

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
#set(CUDA_NVCC_FLAGS -lineinfo;--ptxas-options=-v;-arch=compute_89;-code=sm_89;-maxrregcount 64)
file(GLOB_RECURSE files  "./lib/*.cpp" "./io/*.cpp" "./equation/*.cpp"  "./main.cpp" )

#set(CMAKE_CXX_FLAGS -g)


add_definitions(-DZSPH_VTK)
add_definitions(-DZSPH_ASCII)
add_definitions(-DZSPH_OUTPUT_ACC)
add_definitions(-DZSPH_OUTPUT_VEL)
add_definitions(-DZSPH_OUTPUT_TYPE)
add_definitions(-DZSPH_OUTPUT_TENSOR)

add_executable(dof ${files})
target_link_libraries(dof ${VTK_LIBRARIES})
vtk_module_autoinit(
  TARGETS dof
  MODULES ${VTK_LIBRARIES}
)
