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

find_package(OpenGL REQUIRED)

find_package(CUDA REQUIRED)

add_subdirectory(external)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# executable
#--------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

set(SOURCE_FILES_GPU
    src/glad.c
    src/main.cpp
    src/img.cpp
    src/Camera.cpp
    src/Light.cpp
    src/ShaderProgram.cpp
    src/GLError.cpp
    src/App.cpp
    src/ray_marching/ray_marching.cu
    src/ray_marching/nn_weights.cpp)

cuda_add_executable(main ${SOURCE_FILES_GPU})

include_directories(SYSTEM
    ${STB_INCLUDE_DIRS})
target_include_directories(main
    PRIVATE
    include
    ${GLAD_INCLUDE_DIRS}
    ${CUDA_INCLUDE_DIRS}
    ${OPENGL_INCLUDE_DIR}})

target_link_libraries(main
    glfw
    glm
    ${OPENGL_gl_LIBRARY}
    nlohmann_json)

target_compile_options(main PRIVATE -Wall -Wextra -O3)
