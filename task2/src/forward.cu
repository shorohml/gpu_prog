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

// ----------------------------------------------------------------------------------------------------
// ----------------------------------------------------------------------------------------------------

enum Activation {
    LeakyReLU,
    Tanh,
};

struct AABBOX {

    float3 min;
    float3 max;
    float eps = 1e-4;

    AABBOX(float3 _min, float3 _max)
    {
        min = _min;
        max = _max;
    }

    __device__ float intersect(float3 orig, float3 dir)
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
};

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
                out = alpha * out;
            }
            break;
        case Activation::Tanh:
            out = tanh(out);
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
            out = alpha * out;
        }
        break;
    case Activation::Tanh:
        out = tanh(out);
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
                out = alpha * out;
            }
            break;
        case Activation::Tanh:
            out = tanh(out);
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

__global__ void sphere_tracing(
    float *X_inner_1,
    float *X_inner_2,
    float **weights,
    float *color,
    int *sizes,
    AABBOX bbox,
    CameraCuda cam,
    int W,
    int H,
    float eps)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) {
        return;
    }

    int idx = i * W + j;
    int p_idx = idx * 3;
    int i_idx = idx * 32;
    float d, d_bbox;

    float3 point = cam.pos;

    float nx = ((float)j / W - 0.5) * 2.0;
    float ny = -((float)i / H - 0.5) * 2.0;
    float3 dir = cam.side * nx + cam.up * ny + cam.dir * cam.invhalffov;

    extern __shared__ float weights_s[];

    int t_idx = threadIdx.x * blockDim.y + threadIdx.y;

    int w_idx = 0;
    if (t_idx < 10) {
        for (int i = 0; i < t_idx; ++i) {
            w_idx += sizes[i];
        }
        for (int i = 0; i < sizes[t_idx]; ++i) {
           weights_s[w_idx + i] = weights[t_idx][i]; 
        }
    }

    __syncthreads();

    d_bbox = bbox.intersect(point, dir);
    if (d_bbox == 0.0) {
        color[p_idx] = 1.0;
        color[p_idx + 1] = 1.0;
        color[p_idx + 2] = 1.0;
        return;
    }

    d = forward_point(
        point,
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );

    float total_dist = d;
    bool is_in_bbox = false;
    float3 dir_bbox = dir;
    while (d > eps) {
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
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        total_dist += d;
    }

    float x1 = forward_point(
        make_float3(point.x + 1e-5, point.y, point.z),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float x2 = forward_point(
        make_float3(point.x - 1e-5, point.y, point.z),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float y1 = forward_point(
        make_float3(point.x, point.y + 1e-5, point.z),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float y2 = forward_point(
        make_float3(point.x, point.y - 1e-5, point.z),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float z1 = forward_point(
        make_float3(point.x, point.y, point.z + 1e-5),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float z2 = forward_point(
        make_float3(point.x, point.y, point.z - 1e-5),
        X_inner_1 + i_idx,
        X_inner_2 + i_idx,
        weights_s,
        sizes
    );
    float3 normal = make_float3(
        x1 - x2,
        y1 - y2,
        z1 - z2
    );
    normal = normalize(normal);

    float3 light = normalize(make_float3(0.0, 0.0, 1.0));

    if (dot(light, normal) < 0) {
        normal = -normal;
    }

    d = dot(normal, light);
    float3 color_vec = make_float3(0.1, 0.2, 0.5) * 2.0 * d;

    if (total_dist != -1.0) {
        color[p_idx] = color_vec.x;
        color[p_idx + 1] = color_vec.y;
        color[p_idx + 2] = color_vec.z;
    } else {
        color[p_idx] = 1.0;
        color[p_idx + 1] = 1.0;
        color[p_idx + 2] = 1.0;
    }

    // if (total_dist != -1.0) {
    //     color[p_idx] = normal.x * 0.5 + 0.5;
    //     color[p_idx + 1] = normal.y * 0.5 + 0.5;
    //     color[p_idx + 2] = normal.z * 0.5 + 0.5;
    // } else {
    //     color[p_idx] = 1.0;
    //     color[p_idx + 1] = 1.0;
    //     color[p_idx + 2] = 1.0;
    // }
}


__global__ void sphere_tracing_texture(
    float *X_inner_1,
    float *X_inner_2,
    float **weights,
    cudaSurfaceObject_t surface,
    int *sizes,
    AABBOX bbox,
    CameraCuda cam,
    int W,
    int H,
    float eps)
{
    uint x = blockIdx.x * blockDim.x + threadIdx.x;
    uint y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= H || y >= W) {
        return;
    }

    int idx = x * W + y;
    int p_idx = idx * 3;
    int i_idx = idx * 32;
    float d, d_bbox;

    float3 point = cam.pos;

    float nx = ((float)x / W - 0.5) * 2.0;
    float ny = ((float)y / H - 0.5) * 2.0;
    float3 dir = cam.side * nx + cam.up * ny + cam.dir * cam.invhalffov;

    extern __shared__ float weights_s[];

    int t_idx = threadIdx.x * blockDim.y + threadIdx.y;

    int w_idx = 0;
    if (t_idx < 10) {
        for (int i = 0; i < t_idx; ++i) {
            w_idx += sizes[i];
        }
        for (int i = 0; i < sizes[t_idx]; ++i) {
           weights_s[w_idx + i] = weights[t_idx][i]; 
        }
    }

    __syncthreads();

    float3 color_vec;
    float3 normal;
    float total_dist = d;
    bool is_in_bbox = false;
    float3 dir_bbox = dir;

    d_bbox = bbox.intersect(point, dir);
    if (d_bbox == 0.0) {
        total_dist = -1.0;
        color_vec.x = 1.0;
        color_vec.y = 1.0;
        color_vec.z = 1.0;
    } else {
        d = forward_point(
            point,
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );

        while (d > eps) {
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
                X_inner_1 + i_idx,
                X_inner_2 + i_idx,
                weights_s,
                sizes
            );
            total_dist += d;
        }

        float x1 = forward_point(
            make_float3(point.x + 1e-5, point.y, point.z),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        float x2 = forward_point(
            make_float3(point.x - 1e-5, point.y, point.z),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        float y1 = forward_point(
            make_float3(point.x, point.y + 1e-5, point.z),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        float y2 = forward_point(
            make_float3(point.x, point.y - 1e-5, point.z),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        float z1 = forward_point(
            make_float3(point.x, point.y, point.z + 1e-5),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        float z2 = forward_point(
            make_float3(point.x, point.y, point.z - 1e-5),
            X_inner_1 + i_idx,
            X_inner_2 + i_idx,
            weights_s,
            sizes
        );
        normal = make_float3(
            x1 - x2,
            y1 - y2,
            z1 - z2
        );
        normal = normalize(normal);

        float3 light = normalize(make_float3(0.0, 0.0, 1.0));

        if (dot(dir, normal) > 0) {
            normal = -normal;
        }

        d = dot(normal, light);
        color_vec = make_float3(0.1, 0.2, 0.5) * 2.0 * d;

        if (total_dist == -1.0) {
            color_vec = make_float3(1.0f, 1.0f, 1.0f);
        }
    }

    // uchar4 data;
    // data.x = (unsigned char)(255 * color_vec.x);
    // data.y = (unsigned char)(255 * color_vec.y);
    // data.z = (unsigned char)(255 * color_vec.z);
    // data.w = 255;

    uchar4 data;

    // if (total_dist != -1.0) {
    //     data.x = (unsigned char)(255 * (normal.x * 0.5 + 0.5));
    //     data.y = (unsigned char)(255 * (normal.y * 0.5 + 0.5));
    //     data.z = (unsigned char)(255 * (normal.z * 0.5 + 0.5));
    //     data.w = 255;
    // } else {
    //     data.x = 255;
    //     data.y = 255;
    //     data.z = 255;
    //     data.w = 255;
    // }

    if (total_dist != -1.0) {
        fprintf("%f\n", )

        data.x = (unsigned char)(255 * total_dist);
        data.y = (unsigned char)(255 * total_dist);
        data.z = (unsigned char)(255 * total_dist);
        data.w = 255;
    } else {
        data.x = 255;
        data.y = 255;
        data.z = 255;
        data.w = 255;
    }

    surf2Dwrite(data, surface, x * sizeof(uchar4), y);

    // if (total_dist != -1.0) {
    //     color[p_idx] = normal.x * 0.5 + 0.5;
    //     color[p_idx + 1] = normal.y * 0.5 + 0.5;
    //     color[p_idx + 2] = normal.z * 0.5 + 0.5;
    // } else {
    //     color[p_idx] = 1.0;
    //     color[p_idx + 1] = 1.0;
    //     color[p_idx + 2] = 1.0;
    // }
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


void forward(
    float *color,
    std::vector<std::vector<float> > &weights,
    int W,
    int H,
    float eps)
{
    cudaEvent_t start_cu_1, stop_cu_1;
    float *color_gpu;
    float *inner1;
    float *inner2;
    int sizes[10];
    int *sizes_gpu;

    sizes[0] = 32 * 3;
    sizes[1] = 32;
    sizes[2] = 32 * 32;
    sizes[3] = 32;
    sizes[4] = 32 * 32;
    sizes[5] = 32;
    sizes[6] = 32 * 32;
    sizes[7] = 32;
    sizes[8] = 32;
    sizes[9] = 1;

	dim3 dimBlock(N_THREADS_X, N_THREADS_Y);
	dim3 dimGrid((H - 1) / N_THREADS_X + 1, (W - 1) / N_THREADS_Y + 1);

    Weights weights_gpu(weights, sizes, 10);

    checkCudaErrors(cudaMalloc((void**)&color_gpu, W * H * 3 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&inner1, W * H * 32 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&inner2, W * H * 32 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&sizes_gpu, 10 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(
        sizes_gpu,
        sizes,
        10 * sizeof(int),
        cudaMemcpyHostToDevice));

    Camera cam(
        glm::vec3(0.f, 0.0f, 1.2f),
        glm::vec3(0.0f, 1.0f, 0.0f),
        YAW,
        PITCH
    );

    AABBOX bbox(
        make_float3(-0.76, -0.76, -0.56),
        make_float3(0.76,  0.76,  0.56)
    );

    int size = 0;
    for (int i = 0; i < 10; ++i) {
        size += sizes[i];
    }

    cudaEventCreate(&start_cu_1);
    cudaEventCreate(&stop_cu_1);
    cudaEventRecord(start_cu_1, 0);

    sphere_tracing<<<dimGrid, dimBlock, size * sizeof(float)>>>(
        inner1,
        inner2,
        weights_gpu.weights_gpu,
        color_gpu,
        sizes_gpu,
        bbox,
        cam.getCudaCamera(),
        W,
        H,
        eps
    );

    checkCudaErrors(cudaDeviceSynchronize());

    float time;
    cudaEventRecord(stop_cu_1, 0);
    cudaEventSynchronize(stop_cu_1);
    cudaEventElapsedTime(&time, start_cu_1, stop_cu_1);

    std::cout << time / 1000 << std::endl;

    checkCudaErrors(cudaMemcpy(
        color,
        color_gpu,
        W * H * 3 * sizeof(float),
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());
}


void forward_surface(
    cudaSurfaceObject_t surface,
    NetworkData &network_data,
    Camera &cam,
    int W,
    int H,
    float eps)
{
	dim3 dimBlock(N_THREADS_X, N_THREADS_Y);
	dim3 dimGrid((H - 1) / N_THREADS_X + 1, (W - 1) / N_THREADS_Y + 1);

    AABBOX bbox(
        make_float3(-0.76, -0.76, -0.56),
        make_float3(0.76,  0.76,  0.56)
    );

    int size = 0;
    for (int i = 0; i < network_data.sizes_cpu.size(); ++i) {
        size += network_data.sizes_cpu[i];
    }

    sphere_tracing_texture<<<dimGrid, dimBlock, size * sizeof(float)>>>(
        network_data.inner1,
        network_data.inner2,
        network_data.weights.weights_gpu,
        surface,
        network_data.sizes,
        bbox,
        cam.getCudaCamera(),
        W,
        H,
        eps
    );

    checkCudaErrors(cudaDeviceSynchronize());
}
