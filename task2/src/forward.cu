#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cmath>
#include <glad/glad.h>
#include <GLFW/glfw3.h>
#include <cuda_gl_interop.h>
#include "forward.h"
#include "Camera.h"

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------
// Math operators for float3
// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------


__host__ __device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__host__ __device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__host__ __device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__host__ __device__ float3 operator/(const float3 &a, const float3 &b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__host__ __device__ float3 operator*(const float3 &a, const float &b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

__host__ __device__ float3 minf3(float3 a, float3 b)
{
    return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}

__host__ __device__ float3 maxf3(float3 a, float3 b)
{
    return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}

__host__ __device__ float minf1(float a, float b)
{
    return a < b ? a : b;
}

__host__ __device__ float maxf1(float a, float b)
{
    return a > b ? a : b;
}

__host__ __device__ float dot(float3 a, float3 b)
{
	return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__ float3 normalize(float3 v)
{
	float invLen = rsqrtf(dot(v, v));
	return v * invLen;
}


__host__ __device__ float clamp(float v, float min_v, float max_v)
{
    return maxf1(minf1(v, max_v), min_v);
}

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------

AABBOX::AABBOX(float3 _min, float3 _max)
{
    min = _min;
    max = _max;
}

//see https://www.scratchapixel.com/lessons/3d-basic-rendering/minimal-ray-tracer-rendering-simple-shapes/ray-box-intersection
__device__ float AABBOX::intersect(float3 &orig, float3 &dir)
{
    float3 tmin = (min - orig) / dir;
    float3 tmax = (max - orig) / dir;

    float3 real_min = minf3(tmin, tmax);
    float3 real_max = maxf3(tmin, tmax);

    float minmax = minf1(minf1(real_max.x, real_max.y), real_max.z);
    float maxmin = maxf1(maxf1(real_min.x, real_min.y), real_min.z);

    if (minmax >= maxmin) {
        return maxmin > eps ? maxmin : 0;
    }
    return 0;
}

__device__ bool AABBOX::is_in(float3 point) {
    return (point.x >= min.x) && (point.x <= max.x) &&
            (point.y >= min.y) && (point.y <= max.y) &&
            (point.z >= min.z) && (point.z <= max.z);
}


__device__ void linear(
    float *x_in,
    float *x_out,
    int d_in,
    int d_out,
    float *weights,
    float *bias,
    Activation activation,
    float alpha)
{
    for (int i = 0; i < d_out; ++i) {
        float out = 0.;
        int row = i * d_in;

        for (int j = 0; j < d_in; ++j) {
            out += x_in[j] * weights[row + j];
        }
        out += bias[i];

        switch (activation) {
        case Activation::LeakyReLU:
            if (out < 0) {
                out *= alpha;
            }
            break;
        case Activation::Tanh:
            out = tanh(out);
            break;
        }

        x_out[i] = out;
    }
}

__device__ float linear_out1(
    float *x_in,
    int d_in,
    float *weights,
    float *bias,
    Activation activation,
    float alpha)
{
    float out = 0.;

    for (int j = 0; j < d_in; ++j) {
        out += x_in[j] * weights[j];
    }
    out += bias[0];

    switch (activation) {
    case Activation::LeakyReLU:
        if (out < 0) {
            out *= alpha;
        }
        break;
    case Activation::Tanh:
        out = tanh(out);
        break;
    }

    return out;
}

__device__ void linear_in3(
    float3 point,
    float *x_out,
    int d_out,
    float *weights,
    float *bias,
    Activation activation,
    float alpha)
{
    for (int i = 0; i < d_out; ++i) {
        float out;
        int row = 3 * i;

        out = point.x * weights[row];
        out += point.y * weights[row + 1];
        out += point.z * weights[row + 2];
        out += bias[i];

        switch (activation) {
        case Activation::LeakyReLU:
            if (out < 0) {
                out *= alpha;
            }
            break;
        case Activation::Tanh:
            out = tanh(out);
            break;
        }

        x_out[i] = out;
    }
}

__device__ float forward_point(
    float3 point,
    float *x_inner_1,
    float *x_inner_2,
    float *weights,
    int *sizes)
{
    linear_in3(
        point,
        x_inner_1,
        32,
        weights,
        weights + sizes[0],
        Activation::LeakyReLU,
        0.1
    );

    float *tmp;
    int idx = sizes[0];
    for (int i = 0; i < 3; ++i) {
        idx += sizes[1 + 2 * i];
        linear(
            x_inner_1,
            x_inner_2,
            32,
            32,
            weights + idx,
            weights + idx + sizes[2 + 2 * i],
            Activation::LeakyReLU,
            0.1
        );
        tmp = x_inner_1;
        x_inner_1 = x_inner_2;
        x_inner_2 = tmp;
        idx += sizes[2 + 2 * i];
    }

    idx += sizes[7];
    return linear_out1(
        x_inner_1,
        32,
        weights + idx,
        weights + idx + sizes[8],
        Activation::Tanh,
        0.1
    );
}

__constant__ float tet_vertices[] = {  
    1, -1, -1,
    -1, -1,  1,
    -1,  1, -1,
    1,  1,  1,
};

__global__ void sphere_tracing_texture(
    float *weights,
    uint *d_out,
    int *sizes,
    int size,
    AABBOX bbox,
    CameraCuda cam,
    const RenderingMode mode,
    float3 light,
    int W,
    int H,
    float eps)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= H || y >= W) {
        return;
    }

    float d, d_bbox;
    float3 point = cam.pos;

    float nx = ((float)x / W - 0.5) * 2.0;
    float ny = ((float)y / H - 0.5) * 2.0;
    float3 dir = cam.side * nx + cam.up * ny + cam.dir * cam.invhalffov;

    extern __shared__ float weights_s[];
    __shared__ float X_inner_1_s[N_THREADS_X * N_THREADS_Y * 32];
    __shared__ float X_inner_2_s[N_THREADS_X * N_THREADS_Y * 32];

    int t_idx = threadIdx.x * blockDim.y + threadIdx.y;

    for (int i = 0; i < size / (blockDim.x * blockDim.y - 1) + 1; ++i) {
        weights_s[t_idx + i * blockDim.x * blockDim.y] = weights[t_idx + i * blockDim.x * blockDim.y];
    }

    __syncthreads();

    float3 color_vec;
    float3 normal;
    float total_dist = 0.0;
    bool is_in_bbox = false;
    float3 dir_bbox = dir;
    bool is_in;

    const int MAX_STEPS = 20;

    d_bbox = bbox.intersect(point, dir);
    is_in = bbox.is_in(point);
    if (!is_in && d_bbox == 0.0) {
        total_dist = -1.0;
        color_vec.x = 1.0;
        color_vec.y = 1.0;
        color_vec.z = 1.0;
    } else {
        point = point + dir * d_bbox;
        total_dist += d_bbox;

        d = forward_point(
            point,
            X_inner_1_s + t_idx * 32,
            X_inner_2_s + t_idx * 32,
            weights_s,
            sizes
        );

        for (int i = 0; i < MAX_STEPS; ++i) {
            total_dist += d;

            if (abs(d) < eps) {
                break;
            }

            point = point + dir * d;

            d_bbox = bbox.intersect(point, dir_bbox);
            if (d_bbox == 0.0) {
                /* entered bbox */
                is_in_bbox = true;
                dir_bbox = -dir_bbox;
            } else if (is_in_bbox) {
                /* out of bbox */
                total_dist = -1.0;
                break;
            }

            d = forward_point(
                point,
                X_inner_1_s + t_idx * 32,
                X_inner_2_s + t_idx * 32,
                weights_s,
                sizes
            );
        }

        if (total_dist == -1.0) {
            color_vec = make_float3(1.0f, 1.0f, 1.0f);
        } else {
            //see https://iquilezles.org/www/articles/normalsSDF/normalsSDF.htm
            float3 tet_point = make_float3(tet_vertices[0], tet_vertices[1], tet_vertices[2]);
            float3 p0 = tet_point * forward_point(
                point + tet_point * 1e-5,
                X_inner_1_s + t_idx * 32,
                X_inner_2_s + t_idx * 32,
                weights_s,
                sizes
            );

            tet_point = make_float3(tet_vertices[3], tet_vertices[4], tet_vertices[5]);
            float3 p1 = tet_point * forward_point(
                point + tet_point * 1e-5,
                X_inner_1_s + t_idx * 32,
                X_inner_2_s + t_idx * 32,
                weights_s,
                sizes
            );

            tet_point = make_float3(tet_vertices[6], tet_vertices[7], tet_vertices[8]);
            float3 p2 = tet_point * forward_point(
                point + tet_point * 1e-5,
                X_inner_1_s + t_idx * 32,
                X_inner_2_s + t_idx * 32,
                weights_s,
                sizes
            );

            tet_point = make_float3(tet_vertices[9], tet_vertices[10], tet_vertices[11]);
            float3 p3 = tet_point * forward_point(
                point + tet_point * 1e-5,
                X_inner_1_s + t_idx * 32,
                X_inner_2_s + t_idx * 32,
                weights_s,
                sizes
            );

            normal = normalize(p0 + p1 + p2 + p3);
            d = dot(normal, -light);
            color_vec = make_float3(0.1, 0.2, 0.5) * 2.0 * d;
        }
    }

    uchar4 data;
    if (total_dist == -1.0) {
        data.x = 255;
        data.y = 255;
        data.z = 255;
        data.w = 255;
    } else {
        switch (mode) {
        case RenderingMode::DEFAULT:
            data.x = 255 * color_vec.x;
            data.y = 255 * color_vec.y;
            data.z = 255 * color_vec.z;
            data.w = 255;
            break;
        case RenderingMode::SHADOW_MAP:
            data.x = 255 * (normal.x * 0.5 + 0.5);
            data.y = 255 * (normal.y * 0.5 + 0.5);
            data.z = 255 * (normal.z * 0.5 + 0.5);
            data.w = 255;
            break;
        case RenderingMode::NORMALS_COLOR:
            total_dist = clamp(total_dist, 0.0f, 1.0f);
            data.x = 255 * total_dist;
            data.y = 255 * total_dist;
            data.z = 255 * total_dist;
            break;
        }
    }
    d_out[y * W + x] = data.w << 24 | data.z << 16 | data.y << 8 | data.x;
}


CameraCuda::CameraCuda(float3 _pos, float3 _dir, float3 _up, float3 _side, float _fov)
{
    pos = _pos;
    dir = normalize(_dir);
    up = normalize(_up);
    side = normalize(_side);
    fov = _fov * (float)M_PI / 180.f;
    invhalffov = 1.0f / std::tan(fov / 2.0f);
}

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

void forward_surface(
    uint *d_out,
    NetworkData &network_data,
    const CameraCuda &cam,
    const RenderingMode mode,
    float3 light_dir,
    int W,
    int H,
    float eps)
{
	dim3 dimBlock(N_THREADS_X, N_THREADS_Y);
	dim3 dimGrid((H - 1) / N_THREADS_X + 1, (W - 1) / N_THREADS_Y + 1);

    AABBOX bbox(
        make_float3(-0.76, -0.8, -0.56),
        make_float3(0.76,  0.8,  0.56)
    );

    int size = 0;
    for (int i = 0; i < network_data.sizes_cpu.size(); ++i) {
        size += network_data.sizes_cpu[i];
    }

    sphere_tracing_texture<<<dimGrid, dimBlock, size * sizeof(float)>>>(
        network_data.weights.all_weights_gpu,
        d_out,
        network_data.sizes,
        size,
        bbox,
        cam,
        mode,
        light_dir,
        W,
        H,
        eps
    );
}
