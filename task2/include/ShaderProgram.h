//ShaderProgram class from https://github.com/v-san/ogl-samples
#pragma once

#include <glm/glm.hpp>
#include <unordered_map>

#include "common.h"

class ShaderProgram {
public:
    ShaderProgram()
        : ProgramObj(-1) {};

    ShaderProgram(const std::unordered_map<GLenum, std::string>& inputShaders);

    virtual ~ShaderProgram() {};

    void Release(); //actual destructor

    virtual void StartUseShader() const;

    virtual void StopUseShader() const;

    GLuint GetProgram() const { return ProgramObj; }

    bool reLink();

    void SetUniform(const std::string& location, float value) const;

    void SetUniform(const std::string& location, double value) const;

    void SetUniform(const std::string& location, int value) const;

    void SetUniform(const std::string& location, unsigned int value) const;

    void SetUniform(const std::string& location, const glm::vec3& value) const;

    void SetUniform(const std::string& location, const glm::vec4 &value) const;

    void SetUniform(const std::string& location, const glm::mat3& value) const;

    void SetUniform(const std::string& location, const glm::mat4& value) const;

    GLuint ProgramObj;

private:
    static GLuint LoadShaderObject(GLenum type, const std::string& filename);
    std::unordered_map<GLenum, GLuint> shaderObjects;
};
