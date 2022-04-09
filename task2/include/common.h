//Macros for OpenGl errors check from https://github.com/v-san/ogl-samples
#pragma once

#include <fstream>
#include <glad/glad.h>
#include <iostream>
#include <string>

#include "GLError.h"

#define GL_CHECK_ERRORS ThrowExceptionOnGLError(__LINE__, __FILE__);

enum RenderingMode {
    DEFAULT = 0,
    SHADOW_MAP,
    NORMALS_COLOR
};
