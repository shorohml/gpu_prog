cmake_minimum_required(VERSION 3.16)
set(PROJECT_NAME MMCG)
project(${PROJECT_NAME} LANGUAGES CUDA CXX)

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

#---------------------------------------------------------------------------------------
# CPU
#---------------------------------------------------------------------------------------

set(SOURCE_FILES_CPU
    src/main_cpu.cpp
    src/cpu/hist_eq.cpp)

add_executable(main_cpu ${SOURCE_FILES_CPU})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS})
target_include_directories(main_cpu
    PRIVATE
    include)

target_link_libraries(main_cpu
    PRIVATE
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_compile_options(main_cpu PRIVATE -Werror -Wall -Wextra -O3)

#---------------------------------------------------------------------------------------
# GPU
#---------------------------------------------------------------------------------------

find_package(CUDA REQUIRED)

enable_language(CUDA)

include(CheckLanguage)
check_language(CUDA)

set(SOURCE_FILES_GPU
    src/gpu/hist_eq.cu
    src/main_gpu.cpp)

cuda_add_executable(main_gpu ${SOURCE_FILES_GPU})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS})
target_include_directories(main_gpu
    PRIVATE
    include)

# target_link_libraries(main_gpu
#    PRIVATE
#    ${CUDA_LIBRARIES}
# )

set_target_properties(main_gpu PROPERTIES
                      CUDA_SEPARABLE_COMPILATION ON)

# set(CUDA_LINK_LIBRARIES_KEYWORD PUBLIC)

target_compile_options(main_gpu PRIVATE -Werror -Wall -Wextra -O3)
