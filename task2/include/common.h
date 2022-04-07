//Macros for OpenGl errors check from https://github.com/v-san/ogl-samples
#pragma once

#include <fstream>
#include <glad/glad.h>
#include <iostream>
#include <string>

#include "GLError.h"

//полезный макрос для проверки ошибок
//в строчке, где он был записан вызывает ThrowExceptionOnGLError, которая при возникновении ошибки opengl
//пишет в консоль номер текущей строки и название исходного файла
//а также тип ошибки
#define GL_CHECK_ERRORS ThrowExceptionOnGLError(__LINE__, __FILE__);
