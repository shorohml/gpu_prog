#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <vector>
#include "common.h"

#define N_THREADS_X 8
#define N_THREADS_Y 8

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

enum Activation {
    LeakyReLU,
    Tanh,
};


struct AABBOX {

    float3 min;
    float3 max;
    float eps = 1e-4;

    AABBOX() {}

    AABBOX(float3 _min, float3 _max);

    __device__ float intersect(float3 &orig, float3 &dir);

    __device__ bool is_in(float3 point);
};

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

struct CameraCuda {
public:
    float3 pos;
    float3 dir;
    float3 up;
    float3 side;
    float fov;
    float invhalffov;

    CameraCuda(float3 _pos, float3 _dir, float3 _up, float3 _side, float fov);
};

void forward_surface(
    uint *d_output,
    NetworkData &network_data,
    const CameraCuda& cam,
    const RenderingMode mode,
    float3 light_dir,
    int W,
    int H,
    float eps);
