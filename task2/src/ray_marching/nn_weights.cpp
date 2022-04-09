#include "ray_marching/nn_weights.h"
#include "common.h"
#include <fstream>
#include <sstream>
#include <cuda_runtime.h>

const std::string WHITESPACE = " \n\r\t\f\v";
 
Weights::Weights(std::vector<std::vector<float>>& _weights_cpu)
{
    for (std::size_t i = 0; i < _weights_cpu.size(); ++i) {
        for (std::size_t j = 0; j < _weights_cpu[i].size(); ++j) {
            all_weights.push_back(_weights_cpu[i][j]);
        }
    }
    checkCudaErrors(cudaMalloc((void**)&all_weights_gpu, all_weights.size() * sizeof(float)));
    checkCudaErrors(cudaMemcpy(
        all_weights_gpu,
        all_weights.data(),
        all_weights.size() * sizeof(float),
        cudaMemcpyHostToDevice));
    n_layers = _weights_cpu.size();
}

Weights::~Weights()
{
    cudaFree(all_weights_gpu);
}

NetworkData::NetworkData(std::vector<std::vector<float>>& _weights_cpu)
    : weights(_weights_cpu)
{
    for (std::size_t i = 0; i < _weights_cpu.size(); ++i) {
        sizes_cpu.push_back(_weights_cpu[i].size());
    }
    checkCudaErrors(cudaMalloc((void**)&sizes, sizes_cpu.size() * sizeof(int)));
    checkCudaErrors(cudaMemcpy(
        sizes,
        sizes_cpu.data(),
        sizes_cpu.size() * sizeof(int),
        cudaMemcpyHostToDevice));
}

NetworkData::~NetworkData()
{
    cudaFree(sizes);
}

std::string ltrim(const std::string &s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}
 
std::string rtrim(const std::string &s)
{
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
 
std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}

std::vector<float> read_weights(const std::string& path)
{
    std::ifstream in_file(path);
    std::vector<float> res;

    if (!in_file.is_open()) {
        throw std::runtime_error("Could not read weights from " + path);
    }

    std::string line;
    while (std::getline(in_file, line)) {
        std::stringstream stream(line);

        std::string val;
        while (std::getline(stream, val, ',')) {
            if (0 == trim(val).size()) {
                continue;
            }
            res.push_back(std::atof(val.c_str()));
        }
    }
    return res;
}
