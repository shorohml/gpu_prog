include(ExternalProject)
include(FetchContent)


#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# building
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# GLFW
set(GLFW_BUILD_DOCS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(glfw)

#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------
# header only
#---------------------------------------------------------------------------------------
#---------------------------------------------------------------------------------------

# glad
set(GLAD_INCLUDE_DIRS external/glad/include PARENT_SCOPE)

# stb
set(STB_INCLUDE_DIRS external/stb PARENT_SCOPE)

#download glm
FetchContent_Declare(
  glm
  URL https://github.com/g-truc/glm/releases/download/0.9.9.8/glm-0.9.9.8.zip
  URL_HASH SHA256=37e2a3d62ea3322e43593c34bae29f57e3e251ea89f4067506c94043769ade4c
  GIT_PROGRESS TRUE
)
FetchContent_GetProperties(glm)
if(NOT glm_POPULATED)
  FetchContent_Populate(glm)
  add_subdirectory(${glm_SOURCE_DIR} ${glm_BINARY_DIR})
endif()

#download nlohmann/json
set(JSON_BuildTests OFF)
FetchContent_Declare(
  json
  URL https://github.com/nlohmann/json/archive/refs/tags/v3.9.1.zip
  URL_HASH SHA256=a88449d68aab8d027c5beefe911ba217f5ffcc0686ae1793d37f3d20698b37c6
  GIT_PROGRESS TRUE
)
FetchContent_GetProperties(json)
if(NOT json_POPULATED)
  FetchContent_Populate(json)
  add_subdirectory(${json_SOURCE_DIR} ${json_BINARY_DIR})
endif()
