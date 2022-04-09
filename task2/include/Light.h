// Camera class based on sample from learnopengl.com
#pragma once

#include <glad/glad.h>
#include <glm/glm.hpp>
#include <glm/gtc/matrix_transform.hpp>
#include <vector>
#include <vector_functions.h>
#include <vector_types.h>
#include "ray_marching/ray_marching.h"
#include "Camera.h"

const float LIGHT_SOURCE_SENSITIVITY = 18.0f;

// An abstract light class that processes input and calculates the corresponding Euler Angles, Vectors and Matrices for use in OpenGL
class DirectionalLight {
public:
    // light Attributes
    glm::vec3 Front;
    glm::vec3 Up;
    glm::vec3 Right;
    glm::vec3 WorldUp;
    // euler Angles
    float Yaw;
    float Pitch;
    // light options
    float MovementSpeed;

    // constructor with vectors
    DirectionalLight(glm::vec3 up = glm::vec3(0.0f, 1.0f, 0.0f), float yaw = YAW, float pitch = PITCH);

    // processes input received from any keyboard-like input system. Accepts input parameter in the form of light defined ENUM (to abstract it from windowing systems)
    void ProcessKeyboard(Camera_Movement direction, float deltaTime, GLboolean constrainPitch = true);

    float3 getDirection() {
        return make_float3(Front.x, Front.y, Front.z);
    }

private:
    // calculates the front vector from the light's (updated) Euler Angles
    void updateVectors();
};
