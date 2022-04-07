#pragma once

#include "Camera.h"
#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>

#define N_THREADS_X 16
#define N_THREADS_Y 16

template <typename T>
inline void check(T result, char const* const func, const char* const file,
    int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<uint>(result), cudaGetErrorName(result), func);
        throw std::runtime_error("");
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

struct Weights {
public:
    float** weights_host;
    float** weights_gpu;
    int n_layers;

    Weights(std::vector<std::vector<float>>& _weights_cpu, int* _sizes, int _n_layers)
    {
        n_layers = _n_layers;
        weights_host = (float**)malloc(_n_layers * sizeof(float*));
        for (int i = 0; i < 10; ++i) {
            checkCudaErrors(cudaMalloc((void**)&weights_host[i], _sizes[i] * sizeof(float)));
            checkCudaErrors(cudaMemcpy(
                weights_host[i],
                _weights_cpu[i].data(),
                _sizes[i] * sizeof(float),
                cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMalloc((void**)&weights_gpu, 10 * sizeof(float*)));
        checkCudaErrors(cudaMemcpy(
            weights_gpu,
            weights_host,
            _n_layers * sizeof(float*),
            cudaMemcpyHostToDevice));
    }

    ~Weights()
    {
        for (int i = 0; i < n_layers; ++i) {
            checkCudaErrors(cudaFree(weights_host[i]));
        }
        checkCudaErrors(cudaFree(weights_gpu));
        free(weights_host);
    }
};

struct NetworkData {
public:
    Weights weights;
    std::vector<int> sizes_cpu;
    float* inner1;
    float* inner2;
    int* sizes;
    int inner_size;
    int W;
    int H;

    NetworkData(std::vector<std::vector<float>>& _weights_cpu, std::vector<int> _sizes, int _inner_size, int _W, int _H)
        : weights(_weights_cpu, _sizes.data(), _sizes.size()), sizes_cpu(_sizes), W(_W), H(_H)
    {
        checkCudaErrors(cudaMalloc((void**)&inner1, _W * _H * _inner_size * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&inner2, _W * _H * _inner_size * sizeof(float)));
        checkCudaErrors(cudaMalloc((void**)&sizes, _sizes.size() * sizeof(int)));
        checkCudaErrors(cudaMemcpy(
            sizes,
            _sizes.data(),
            _sizes.size() * sizeof(int),
            cudaMemcpyHostToDevice));
    }

    ~NetworkData()
    {
        cudaFree(inner1);
        cudaFree(inner2);
        cudaFree(sizes);
    }
};

void forward(
    float* color,
    std::vector<std::vector<float>>& weights,
    int W,
    int H,
    float eps);

void forward_surface(
    cudaSurfaceObject_t surface,
    NetworkData &network_data,
    Camera& cam,
    int W,
    int H,
    float eps);
