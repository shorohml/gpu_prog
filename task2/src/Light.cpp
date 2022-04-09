#include "Light.h"

DirectionalLight::DirectionalLight(glm::vec3 up, float yaw, float pitch)
    : Front(glm::vec3(0.0f, 0.0f, -1.0f))
    , MovementSpeed(LIGHT_SENSITIVITY)
{
    WorldUp = up;
    Yaw = yaw;
    Pitch = pitch;
    updateVectors();
}

void DirectionalLight::ProcessKeyboard(Camera_Movement direction, float deltaTime, GLboolean constrainPitch)
{
    float velocity = MovementSpeed * deltaTime;
    float xoffset = 0.0f;
    float yoffset = 0.0f;

    switch (direction) {
    case FORWARD:
        yoffset = -velocity;
        break;
    case BACKWARD:
        yoffset = velocity;
        break;
    case LEFT:
        xoffset = velocity;
        break;
    case RIGHT:
        xoffset = -velocity;
        break;
    }

    Yaw += xoffset;
    Pitch += yoffset;

    // make sure that when pitch is out of bounds, screen doesn't get flipped
    if (constrainPitch) {
        if (Pitch > 89.0f)
            Pitch = 89.0f;
        if (Pitch < -89.0f)
            Pitch = -89.0f;
    }

    // update Front, Right and Up Vectors using the updated Euler angles
    updateVectors();
}

void DirectionalLight::updateVectors()
{
    // calculate the new Front vector
    glm::vec3 front;
    front.x = cos(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    front.y = sin(glm::radians(Pitch));
    front.z = sin(glm::radians(Yaw)) * cos(glm::radians(Pitch));
    Front = glm::normalize(front);
    // also re-calculate the Right and Up vector
    Right = glm::normalize(glm::cross(Front, WorldUp)); // normalize the vectors, because their length gets closer to 0 the more you look up or down which results in slower movement.
    Up = glm::normalize(glm::cross(Right, Front));
}
