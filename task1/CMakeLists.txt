cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME GPU_PROG_TASK_1)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 11)

set(CMAKE_BUILD_TYPE Release)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# dependencies
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

add_subdirectory(external)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# executable
#--------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

find_package(CUDA REQUIRED)

set(SOURCE_FILES_GPU
    src/main.cpp
    src/img.cpp
    src/cpu/hist_eq.cpp
    src/gpu/hist_eq.cu)

cuda_add_executable(main ${SOURCE_FILES_GPU})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS})
target_include_directories(main
    PRIVATE
    include
    ${CUDA_INCLUDE_DIRS})

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_compile_options(main PRIVATE -Werror -Wall -Wextra)
