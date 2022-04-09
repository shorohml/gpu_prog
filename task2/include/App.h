//Application class
#pragma once

#include "Camera.h"
#include "ShaderProgram.h"
#include "forward.h"
#include "Light.h"

#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <memory>
#include <nlohmann/json.hpp>
#include <unordered_map>
#include <vector>
#include <cuda_gl_interop.h>

struct AppState {
public:
    float deltaTime = 0.0f; //Time between current frame and last frame
    float lastFrame = 0.0f; //Time of last frame
    float lastX = 128, lastY = 128; //Last cursor position
    bool firstMouse = true; //true if mouse didn't move
    int filling = 1; //каркасный режим или нет
    std::vector<bool> keys; //массив состояний кнопок - нажата/не нажата
    bool g_captureMouse = true; //Мышка захвачена нашим приложением или нет?
    RenderingMode renderingMode = RenderingMode::DEFAULT;
    Camera camera; //camera
    DirectionalLight light;

    AppState()
        : keys(1024, 0)
        , camera(glm::vec3(0.f, 0.0f, 1.7f), glm::vec3(0.0f, 1.0f, 0.0f), YAW, PITCH)
    {
    }
};

class App {
public:
    App(const std::string& pathToConfig, std::vector<std::vector<float> > &_weights);

    App(const App&) = delete;

    App& operator=(const App& other) = delete;

    ~App() { release(); }

    int Run();

private:
    AppState state; //we need to separate this into struct because of C-style GLFW callbacks

    nlohmann::json config; //application config
    GLFWwindow* window; //window
    float printEvery = 1.0f;
    std::string shadersPath;

    //nn weigths
    NetworkData network_data;

    //color buffer
    GLuint pbo = 0;     // OpenGL pixel buffer object
    GLuint tex = 0;     // OpenGL texture object
    struct cudaGraphicsResource *cuda_pbo_resource;

    void setupColorBuffer();
    void deleteColorBuffer();

    //simple quad that fills screen
    GLuint quadVAO;
    GLuint quadVBO;
    GLuint quadEBO;
    void setupQuad();
    void deleteQuad();
    void visualizeScene(ShaderProgram& quadColorProgram);

    int initGL() const;

    // callbacks
    static void OnKeyboardPressed(GLFWwindow* window, int key, int /* scancode */, int action, int /* mode */);
    static void OnMouseButtonClicked(GLFWwindow* window, int button, int action, int /* mods */);
    static void OnMouseMove(GLFWwindow* window, double xpos, double ypos);
    static void OnMouseScroll(GLFWwindow* window, double /* xoffset */, double yoffset);

    //move camera
    void doCameraMovement();

    //move light dir
    void doLightMovement();

    //create GLFW window
    int createWindow();

    //main application loop
    void mainLoop();

    //release GPU resources
    void release();
};