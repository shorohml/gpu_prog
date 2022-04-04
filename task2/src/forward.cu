#include <cuda_runtime.h>
#include <vector_types.h>
#include <vector_functions.h>
#include <algorithm>
#include <iostream>
#include "forward.h"

enum Activation {
    LeakyReLU,
    Tanh,
};

struct Weights {
public:
    float **weights_host;
    float **weights_gpu;
    int n_layers;

    Weights(float **_weights_cpu, int *_sizes, int _n_layers) {
        n_layers = _n_layers;
        weights_host = (float **)malloc(_n_layers * sizeof(float *));
        for (int i = 0; i < 10; ++i) {
            checkCudaErrors(cudaMalloc((void**)&weights_host[i], _sizes[i] * sizeof(float)));
            checkCudaErrors(cudaMemcpy(
                weights_host[i],
                _weights_cpu[i],
                _sizes[i] * sizeof(float),
                cudaMemcpyHostToDevice));
        }
        checkCudaErrors(cudaMalloc((void**)&weights_gpu, 10 * sizeof(float *)));
        checkCudaErrors(cudaMemcpy(
            weights_gpu,
            weights_host,
            _n_layers * sizeof(float *),
            cudaMemcpyHostToDevice));
    }

    ~Weights() {
        for (int i = 0; i < n_layers; ++i) {
            checkCudaErrors(cudaFree(weights_host[i]));
        }
        checkCudaErrors(cudaFree(weights_gpu));
        free(weights_host);
    }
};

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// Math operators for float3
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------


__device__ float3 operator+(const float3 &a, const float3 &b)
{
    return make_float3(a.x + b.x, a.y + b.y, a.z + b.z);
}

__device__ float3 operator-(const float3 &a, const float3 &b)
{
    return make_float3(a.x - b.x, a.y - b.y, a.z - b.z);
}

__device__ float3 operator-(const float3 &a)
{
    return make_float3(-a.x, -a.y, -a.z);
}

__device__ float3 operator/(const float3 &a, const float3 &b)
{
    return make_float3(a.x / b.x, a.y / b.y, a.z / b.z);
}

__device__ float3 operator*(const float3 &a, const float &b)
{
    return make_float3(a.x * b, a.y * b, a.z * b);
}

inline __device__ float3 minf3(float3 a, float3 b)
{
    return make_float3(a.x < b.x ? a.x : b.x, a.y < b.y ? a.y : b.y, a.z < b.z ? a.z : b.z);
}

inline __device__ float3 maxf3(float3 a, float3 b)
{
    return make_float3(a.x > b.x ? a.x : b.x, a.y > b.y ? a.y : b.y, a.z > b.z ? a.z : b.z);
}

inline __device__ float minf1(float a, float b)
{
    return a < b ? a : b;
}

inline __device__ float maxf1(float a, float b)
{
    return a > b ? a : b;
}

// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------
// --------------------------------------------------------------------------------------

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
    float *P,
    float *D,
    float *dist,
    float *X_inner_1,
    float *X_inner_2,
    float **weights,
    int *sizes,
    AABBOX bbox,
    int W,
    int H,
    float eps)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    uint j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= H || j >= W) {
        return;
    }

    extern __shared__ float weights_s[];

    int idx = i * W + j;
    int p_idx = idx * 3;
    int i_idx = idx * 32;
    float d, d_bbox;

    int t_idx = threadIdx.x * blockDim.y + threadIdx.y;

    int w_idx = 0;
    for (int i = 0; i < 10; ++i) {
        if (t_idx < sizes[i]) {
           weights_s[w_idx + t_idx] = weights[i][t_idx]; 
        }
        w_idx += sizes[i];
    }
    __syncthreads();

    float3 dir = make_float3(
        D[p_idx],
        D[p_idx + 1],
        D[p_idx + 2]);
    float3 point = make_float3(
        P[p_idx],
        P[p_idx + 1],
        P[p_idx + 2]);

    d_bbox = bbox.intersect(point, dir);
    if (d_bbox == 0.0) {
        dist[idx] = 1.0;
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
    while (abs(d) > eps) {
        point = point + dir * d;

        d_bbox = bbox.intersect(point, dir_bbox);
        if (d_bbox == 0.0) {
            /* entered bbox */
            is_in_bbox = true;
            dir_bbox = -dir_bbox;
        } else if (is_in_bbox) {
            /* out of bbox */
            dist[idx] = 1.0;
            return;
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
    dist[idx] = total_dist;
}

void forward(
    float *in,
    float *out,
    float **weights,
    int W,
    int H,
    float eps)
{
    cudaEvent_t start_cu_1, stop_cu_1;
    float *in_gpu;
    float *out_gpu;
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

    Weights weights_gpu(weights, sizes, 10);

    checkCudaErrors(cudaMalloc((void**)&in_gpu, W * H * 3 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&out_gpu, W * H * sizeof(float)));
    checkCudaErrors(cudaMemcpy(
        in_gpu,
        in,
        W * H * 3 * sizeof(float),
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaMalloc((void**)&inner1, W * H * 32 * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&inner2, W * H * 32 * sizeof(float)));

    checkCudaErrors(cudaMalloc((void**)&sizes_gpu, 10 * sizeof(int)));
    checkCudaErrors(cudaMemcpy(
        sizes_gpu,
        sizes,
        10 * sizeof(int),
        cudaMemcpyHostToDevice));

	dim3 dimBlock(32, 32);
	dim3 dimGrid((H - 1) / 32 + 1, (W - 1) / 32 + 1);

    float *D = (float *)malloc(W * H * 3 * sizeof(float));
    for (int i = 0; i < W * H; ++i) {
        D[3 * i] = 0.0;
        D[3 * i + 1] = 0.;
        D[3 * i + 2] = -1.0;
    }
    float *D_gpu;
    checkCudaErrors(cudaMalloc((void**)&D_gpu, W * H * 3 * sizeof(float)));
    checkCudaErrors(cudaMemcpy(
        D_gpu,
        D,
        W * H * 3 * sizeof(float),
        cudaMemcpyHostToDevice));

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
        in_gpu,
        D_gpu,
        out_gpu,
        inner1,
        inner2,
        weights_gpu.weights_gpu,
        sizes_gpu,
        bbox,
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
        out,
        out_gpu,
        W * H * sizeof(float),
        cudaMemcpyDeviceToHost));
    checkCudaErrors(cudaDeviceSynchronize());
}
