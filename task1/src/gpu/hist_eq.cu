#include <cuda_runtime.h>
#include "gpu/hist_eq.h"


__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}


__global__ void rgb_to_ycbcr(uchar3 *rgb, uchar *y_img, uchar *cb_img, uchar *cr_img, int size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    unsigned char r = rgb[i].x;
    unsigned char g = rgb[i].y;
    unsigned char b = rgb[i].z;

    y_img[i] = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
    cb_img[i] = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
    cr_img[i] = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;
}


__global__ void ycbcr_to_rgb(uchar *y_img, uchar *cb_img, uchar *cr_img, uchar3 *rgb, int size)
{
    uint i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= size) {
        return;
    }

    float y = (float)y_img[i] - 16.;
    float cb = (float)cb_img[i] - 128.;
    float cr = (float)cr_img[i] - 128.;

    float r = 1.164 * y + 1.596 * cr;
    float g = 1.164 * y - 0.392 * cb - 0.813 * cr;
    float b = 1.164 * y + 2.017 * cb;

    rgb[i].x = clamp(r, 0., 255.);
    rgb[i].y = clamp(g, 0., 255.);
    rgb[i].z = clamp(b, 0., 255.);
}


__global__ void histogram(uchar *y_img, uint *hist, int size)
{
    __shared__ uchar shared_hist[SHARED_S];

    for (uint i = 0; i < HIST_SIZE / 4; ++i) {
        ((uint *)shared_hist)[threadIdx.x * HIST_SIZE / 4 + i] = 0;
    }

    __syncthreads();

    uchar *thread_hist = shared_hist + threadIdx.x * HIST_SIZE;

    for (uint pos = blockIdx.x * blockDim.x + threadIdx.x; pos < size; pos += blockDim.x * gridDim.x) {
        uchar y = y_img[pos];
        thread_hist[y] += 1;
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


void equalize_histogram(
        uchar *rgb_cpu,
        int width,
        int height)
{
    uint *hist_gpu, *block_hist_gpu;
    uchar3 *rgb_gpu;
    uchar *y_gpu, *cb_gpu, *cr_gpu;
    float *cdf_gpu;

	dim3 dimBlock(N_THREADS_PER_BLOCK);
	dim3 dimGrid(((width * height) - 1) / N_THREADS_PER_BLOCK + 1);
	dim3 dimBlockHist(HIST_SIZE);
	dim3 dimGridHist(1);
	dim3 dimBlockH(N_THREADS_PER_BLOCK);
	dim3 dimGridH(((width * height) - 1) / (N_THREADS_PER_BLOCK * (HIST_SIZE - 1)) + 1);

    checkCudaErrors(cudaMalloc((void**)&rgb_gpu, width * height * 3));
    checkCudaErrors(cudaMalloc((void**)&y_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cb_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cr_gpu, width * height));
    checkCudaErrors(cudaMalloc((void**)&cdf_gpu, HIST_SIZE * sizeof(float)));
    checkCudaErrors(cudaMalloc((void**)&hist_gpu, HIST_SIZE * sizeof(uint)));
    checkCudaErrors(cudaMalloc((void **)&block_hist_gpu, dimGridH.x * HIST_SIZE * sizeof(uint)));

    checkCudaErrors(cudaMemcpy(rgb_gpu, rgb_cpu, width * height * 3, cudaMemcpyHostToDevice));

    rgb_to_ycbcr<<<dimGrid, dimBlock>>>(rgb_gpu, y_gpu, cb_gpu, cr_gpu, width * height);

    histogram<<<dimGridH, dimBlockH>>>(y_gpu, block_hist_gpu, width * height);

    merge_block_histograms<<<HIST_SIZE, MERGE_THREADBLOCK_SIZE>>>(hist_gpu, block_hist_gpu, dimGridH.x);

    calcCDF<<<dimGridHist, dimBlockHist>>>(cdf_gpu, hist_gpu, width * height);

    equalize<<<dimGrid, dimBlock>>>(y_gpu, cdf_gpu, width * height);

    ycbcr_to_rgb<<<dimGrid, dimBlock>>>(y_gpu, cb_gpu, cr_gpu, rgb_gpu, width * height);

    cudaMemcpy(rgb_cpu, rgb_gpu, width * height * 3, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaFree(rgb_gpu));
    checkCudaErrors(cudaFree(y_gpu));
    checkCudaErrors(cudaFree(cb_gpu));
    checkCudaErrors(cudaFree(cr_gpu));
    checkCudaErrors(cudaFree(cdf_gpu));
    checkCudaErrors(cudaFree(hist_gpu));
    checkCudaErrors(cudaFree(block_hist_gpu));
}
