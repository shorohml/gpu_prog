cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME GPU_PROG_TASK_1)
project(${PROJECT_NAME})

set(CMAKE_CXX_STANDARD 14)

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

find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES_GPU
    src/main.cpp
    src/img.cpp
    src/forward.cu)

cuda_add_executable(main ${SOURCE_FILES_GPU})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS})
target_include_directories(main
    PRIVATE
    include
    ${CUDA_INCLUDE_DIRS}
    ${TORCH_INCLUDE_DIRS})

target_link_libraries(main "${TORCH_LIBRARIES}")

target_compile_options(main PRIVATE -Werror -Wall -Wextra)
