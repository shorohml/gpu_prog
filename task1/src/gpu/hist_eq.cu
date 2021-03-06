#include <cuda_runtime.h>
#include <iostream>
#include "gpu/hist_eq.h"

template <typename T>
__device__ T clamp(T x, T a, T b)
{
    return max(a, min(b, x));
}


__global__ void rgb_to_ycbcr(uchar3 *rgb, uchar *y_img, uchar *cb_img, uchar *cr_img, int size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    int r = rgb[i].x;
    int g = rgb[i].y;
    int b = rgb[i].z;

    y_img[i] = (257 * r + 504 * g + 98 * b) / 1000 + 16;
    cb_img[i] = (-148 * r - 291 * g + 439 * b) / 1000 + 128;
    cr_img[i] = (439 * r - 368 * g - 71 * b) / 1000 + 128;
}


__global__ void ycbcr_to_rgb(uchar *y_img, uchar *cb_img, uchar *cr_img, uchar3 *rgb, int size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    int y = (int)y_img[i] - 16;
    int cb = (int)cb_img[i] - 128;
    int cr = (int)cr_img[i] - 128;

    rgb[i].x = clamp((1164 * y + 1596 * cr) / 1000, 0, 255);
    rgb[i].y = clamp((1164 * y - 392 * cb - 813 * cr) / 1000, 0, 255);
    rgb[i].z = clamp((1164 * y + 2017 * cb) / 1000, 0, 255);
}


__global__ void histogram(uchar *y_img, uint *hist, int size)
{ 
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }
    atomicAdd(hist + y_img[i], 1);
}


__global__ void block_histograms(uchar *y_img, uint *hist, int size)
{
    __shared__ uchar shared_hist[SHARED_S];

    for (uint i = 0; i < HIST_SIZE / 4; ++i) {
        ((uint *)shared_hist)[threadIdx.x * HIST_SIZE / 4 + i] = 0;
    }

    __syncthreads();

    uchar *thread_hist = shared_hist + threadIdx.x * HIST_SIZE;

    for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
        uchar y = y_img[pos];
        ++thread_hist[y];
    }

    __syncthreads();

    for (int idx = threadIdx.x; idx < HIST_SIZE; idx += N_THREADS_PER_BLOCK) {
        uchar *thread_shared_hist = shared_hist + idx;

        uint sum = 0;
        for (uint i = 0; i < SHARED_S; i += HIST_SIZE) {
            sum += thread_shared_hist[i];
        }

        hist[blockIdx.x * HIST_SIZE + idx] = sum;
    }
}


__global__ void merge_block_histograms(uint *hist,
                                       uint *block_hist,
                                       uint histogramCount) {
    __shared__ uint data[MERGE_THREADBLOCK_SIZE];

    uint sum = 0;

    for (uint i = threadIdx.x; i < histogramCount; i += MERGE_THREADBLOCK_SIZE) {
        sum += block_hist[blockIdx.x + i * HIST_SIZE];
    }

    data[threadIdx.x] = sum;

    for (uint stride = MERGE_THREADBLOCK_SIZE / 2; stride > 0; stride >>= 1) {
        __syncthreads();

        if (threadIdx.x < stride) {
            data[threadIdx.x] += data[threadIdx.x + stride];
        }
    }

    if (threadIdx.x == 0) {
        hist[blockIdx.x] = data[0];
    }
}


__global__ void calcCDF(
        float *cdf,
        uint *hist,
	    int size) {
	__shared__ float cdf_shared[HIST_SIZE];
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < HIST_SIZE) {
        cdf_shared[i] = (float)hist[i] / (float)(size);
    }

    __syncthreads();

	for (uint stride = 1; stride <= HIST_SIZE; stride *= 2) {
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < HIST_SIZE)
			cdf_shared[index] += cdf_shared[index - stride];
		__syncthreads();
	}

	for (uint stride = HIST_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < HIST_SIZE) {
			cdf_shared[index + stride] += cdf_shared[index];
		}
	}

	__syncthreads();

	if (i < HIST_SIZE) {
		cdf[i] = cdf_shared[threadIdx.x];
    }
}


__global__ void equalize(uchar *y_img, float *cdf, int size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    y_img[i] = 219 * cdf[y_img[i]] + 16;
}


HistData::HistData(const int width, const int height, const int channels)
{
    uint n_blocks = ((width * height) - 1) / (N_THREADS_PER_BLOCK * (HIST_SIZE - 1)) + 1;
    checkCudaErrors(cudaMalloc((void**)&rgb_gpu, width * height * channels));
    checkCudaErrors(cudaMalloc((void**)&y_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cb_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cr_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cdf_gpu, HIST_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&hist_gpu, HIST_SIZE * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void**)&block_hist_gpu, n_blocks * HIST_SIZE * sizeof(uint)));
}


HistData::~HistData()
{
    checkCudaErrors(cudaFree(rgb_gpu));
    checkCudaErrors(cudaFree(y_gpu));
    checkCudaErrors(cudaFree(cb_gpu));
    checkCudaErrors(cudaFree(cr_gpu));
    checkCudaErrors(cudaFree(cdf_gpu));
    checkCudaErrors(cudaFree(hist_gpu));
    checkCudaErrors(cudaFree(block_hist_gpu));
}


void EqualizeHistogramGPU::process(Img& rgb_img)
{
    cudaEvent_t start_cu_1, stop_cu_1;
    cudaEvent_t start_cu_2, stop_cu_2;

    const int width = rgb_img.get_width();
    const int height = rgb_img.get_height();
    const int channels = rgb_img.get_channels();

	dim3 dimBlock(N_THREADS_PER_BLOCK);
	dim3 dimGrid(((width * height) - 1) / N_THREADS_PER_BLOCK + 1);
	dim3 dimBlockHist(HIST_SIZE);
	dim3 dimGridHist(1);
	dim3 dimBlockH(N_THREADS_PER_BLOCK);
	dim3 dimGridH(((width * height) - 1) / (N_THREADS_PER_BLOCK * (HIST_SIZE - 1)) + 1);

    HistData hist_data(width, height, channels);

    cudaEventCreate(&start_cu_1);
    cudaEventCreate(&stop_cu_1);
    cudaEventRecord(start_cu_1, 0);

    checkCudaErrors(cudaMemcpy(
        hist_data.rgb_gpu,
        rgb_img.get_data(),
        width * height * 3,
        cudaMemcpyHostToDevice));

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventCreate(&start_cu_2);
    cudaEventCreate(&stop_cu_2);
    cudaEventRecord(start_cu_2, 0);

    rgb_to_ycbcr<<<dimGrid, dimBlock>>>(
        hist_data.rgb_gpu,
        hist_data.y_gpu,
        hist_data.cb_gpu,
        hist_data.cr_gpu,
        width * height);
    switch (_mode) {
    case HistComputationMode::DEFAULT:
        histogram<<<dimGrid, dimBlock>>>(
            hist_data.y_gpu,
            hist_data.hist_gpu,
            width * height);
        break;
    case HistComputationMode::SEPARATE_THREAD_HIST:
        block_histograms<<<dimGridH, dimBlockH>>>(
            hist_data.y_gpu,
            hist_data.block_hist_gpu,
            width * height);
        merge_block_histograms<<<HIST_SIZE, MERGE_THREADBLOCK_SIZE>>>(
            hist_data.hist_gpu,
            hist_data.block_hist_gpu,
            dimGridH.x);
        break;
    }
    calcCDF<<<dimGridHist, dimBlockHist>>>(
        hist_data.cdf_gpu,
        hist_data.hist_gpu,
        width * height);
    equalize<<<dimGrid, dimBlock>>>(
        hist_data.y_gpu,
        hist_data.cdf_gpu,
        width * height);
    ycbcr_to_rgb<<<dimGrid, dimBlock>>>(
        hist_data.y_gpu,
        hist_data.cb_gpu,
        hist_data.cr_gpu,
        hist_data.rgb_gpu,
        width * height);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(stop_cu_2, 0);
    cudaEventSynchronize(stop_cu_2);
    cudaEventElapsedTime(&_time_wo_copy, start_cu_2, stop_cu_2);

    checkCudaErrors(cudaMemcpy(
        rgb_img.get_data(),
        hist_data.rgb_gpu,
        width * height * 3,
        cudaMemcpyDeviceToHost));

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(stop_cu_1, 0);
    cudaEventSynchronize(stop_cu_1);
    cudaEventElapsedTime(&_time_w_copy, start_cu_1, stop_cu_1);
}
