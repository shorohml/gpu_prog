cmake_minimum_required(VERSION 3.14)
set(PROJECT_NAME MMCG)
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

set(SOURCE_FILES
    src/main.cpp
    src/hist_eq.cpp)

add_executable(main ${SOURCE_FILES})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS})
target_include_directories(main
    PRIVATE
    include)

target_link_libraries(main
    PRIVATE
)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
    set (CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
endif()

target_compile_options(main PRIVATE -Werror -Wall -Wextra -O3)