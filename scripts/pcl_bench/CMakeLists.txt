cmake_minimum_required(VERSION 3.16)
project(pcl_bench CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()

# Ubuntu's VTK 9.1 (pulled in transitively by PCL's io component) declares an
# MPI link dependency, so the MPI imported targets must exist before PCL is
# resolved — otherwise CMake fails on "VTK::mpi contains MPI::MPI_C but the
# target was not found". We try real MPI first and otherwise provide a harmless
# stub: this benchmark uses only PCL's CPU algorithms, never MPI.
find_package(MPI QUIET COMPONENTS C CXX)
foreach(mpi_target MPI::MPI_C MPI::MPI_CXX)
  if(NOT TARGET ${mpi_target})
    add_library(${mpi_target} INTERFACE IMPORTED)
  endif()
endforeach()

find_package(PCL REQUIRED COMPONENTS common io filters features registration search)

add_executable(pcl_bench pcl_bench.cpp)
target_include_directories(pcl_bench PRIVATE ${PCL_INCLUDE_DIRS})
target_link_libraries(pcl_bench PRIVATE ${PCL_LIBRARIES})
target_compile_definitions(pcl_bench PRIVATE ${PCL_DEFINITIONS})
