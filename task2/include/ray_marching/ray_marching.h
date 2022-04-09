#pragma once

#include <stdexcept>
#include <vector>
#include "common.h"
#include "ray_marching/nn_weights.h"
#include "ray_marching/cuda_utils.h"

#define N_THREADS_X 8
#define N_THREADS_Y 8
#define MAX_STEPS 20
#define HIDDEN_SIZE 32

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

void render_w_ray_marhing(
    uint* d_output,
    NetworkData& network_data,
    const AABBOX &bbox,
    const CameraCuda& cam,
    const RenderingMode& mode,
    const float3& light_dir,
    const int W,
    const int H,
    const float eps);
