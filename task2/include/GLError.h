//Function for OpenGl errors check from https://github.com/v-san/ogl-samples
#pragma once

#include <fstream>
#include <glad/glad.h>
#include <iostream>
#include <string>

void ThrowExceptionOnGLError(int line, const char* file);
