#pragma once

#include "ShaderProgram.h"
#include <glm/glm.hpp>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

struct AABBOX {
    glm::vec3 min;
    glm::vec3 max;

    float GetVolume() const
    {
        glm::vec3 diff = max - min;
        return diff.x * diff.y * diff.z;
    }
};

struct Mesh {
public:
    std::vector<glm::vec3> positions;
    std::vector<glm::vec3> normals;
    std::vector<glm::vec2> texCoords;
    std::vector<uint32_t> indices;
    std::uint32_t matId;
    glm::mat4 model;
    bool isStatic;
    bool hasTangentsBitangents;
    std::vector<glm::vec3> tangents;
    std::vector<glm::vec3> bitangents;
    std::string name;
    bool isEmpty;

    Mesh()
        : isEmpty(true)
    {
    }

    Mesh(
        std::vector<glm::vec3>& positions_,
        std::vector<glm::vec3>& normals_,
        std::vector<glm::vec2>& texCoords_,
        std::vector<uint32_t>& indices_,
        uint32_t matId_ = 0,
        glm::mat4 model_ = glm::mat4(1.0f),
        bool isStatic_ = true)
        : positions(std::move(positions_))
        , normals(std::move(normals_))
        , texCoords(std::move(texCoords_))
        , indices(std::move(indices_))
        , matId(matId_)
        , model(model_)
        , isStatic(isStatic_)
        , hasTangentsBitangents(false)
        , isEmpty(false)
    {
    }

    Mesh(
        std::vector<glm::vec3>& positions_,
        std::vector<glm::vec3>& normals_,
        std::vector<glm::vec2>& texCoords_,
        std::vector<uint32_t>& indices_,
        std::vector<glm::vec3>& tangents_,
        std::vector<glm::vec3>& bitangents_,
        uint32_t matId_ = 0,
        glm::mat4 model_ = glm::mat4(1.0f),
        bool isStatic_ = true)
        : positions(std::move(positions_))
        , normals(std::move(normals_))
        , texCoords(std::move(texCoords_))
        , indices(std::move(indices_))
        , matId(matId_)
        , model(model_)
        , isStatic(isStatic_)
        , hasTangentsBitangents(true)
        , tangents(std::move(tangents_))
        , bitangents(std::move(bitangents_))
        , isEmpty(false)
    {
    }

    std::uint32_t numberOfVertices() const
    {
        return positions.size();
    }

    std::uint32_t numberOfFaces() const
    {
        return indices.size() / 3;
    }

    void GLLoad();
    void GLUpdatePositionsNormals();

    void Draw() const;
    void Draw(const std::vector<glm::mat4>& modelMatrices) const;

    void Release();

    bool IsLoaded() const
    {
        return isLoaded;
    }

    AABBOX GetAABBOX(const bool inWorldSpace = true) const;

private:
    bool isLoaded = false;
    GLuint positionsVBO;
    GLuint normalsVBO;
    GLuint texCoordsVBO;
    GLuint modelsVBO;
    GLuint tangentsVBO;
    GLuint bitangentsVBO;
    GLuint VAO;
    GLuint EBO;
};

std::unique_ptr<Mesh> createCube();