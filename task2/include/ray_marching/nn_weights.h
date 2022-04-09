#pragma once

#include "ray_marching/cuda_utils.h"
#include <vector>
#include <string>

struct Weights {
public:
    int n_layers;
    std::vector<float> all_weights;
    float *all_weights_gpu;

    Weights(std::vector<std::vector<float>>& _weights_cpu);

    ~Weights();
};

struct NetworkData {
public:
    Weights weights;
    std::vector<int> sizes_cpu;
    int* sizes;

    NetworkData(std::vector<std::vector<float>>& _weights_cpu);

    ~NetworkData();
};

std::string ltrim(const std::string &s);

std::string rtrim(const std::string &s);

std::string trim(const std::string &s);

std::vector<float> read_weights(const std::string& path);
