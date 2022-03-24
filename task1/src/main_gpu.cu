#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <cuda_runtime.h>
#include <stb_image_write.h>


#define NUM_WARPS 32
#define WARP_SIZE 32
#define THREADS_IN_BLOCK NUM_WARPS * WARP_SIZE
#define SHARED_MEM_SIZE NUM_WARPS * 256

#define HISTOGRAM_LENGTH 256
#define BLOCK_SIZE 256
#define SCAN_SIZE (2*BLOCK_SIZE)

#define uchar unsigned char


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


#define N_THREADS_PER_BLOCK 192
#define HIST_SIZE 256
#define SHARED_S N_THREADS_PER_BLOCK * HIST_SIZE
#define SHARED_MEMORY_BANKS 16
#define MERGE_THREADBLOCK_SIZE 256


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
	__shared__ float cdf_shared[256];
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < 256) {
        cdf_shared[i] = (float)hist[i] / (float)(size);
    }

    __syncthreads();

	for (uint stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 256)
			cdf_shared[index] += cdf_shared[index - stride];
		__syncthreads();
	}

	for (uint stride = BLOCK_SIZE / 2; stride > 0; stride /= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index + stride < 256) {
			cdf_shared[index + stride] += cdf_shared[index];
		}
	}

	__syncthreads();

	if (i < 256) {
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


static const char *_cudaGetErrorEnum(cudaError_t error) {
    return cudaGetErrorName(error);
}


template <typename T>
void check(T result, char const *const func, const char *const file,
           int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
                static_cast<uint>(result), _cudaGetErrorEnum(result), func);
        exit(EXIT_FAILURE);
    }
}


#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)


int main(int argc, char** argv)
{
    int width, height, channels;
    unsigned char *rgb_img;
    uchar3 *rgb_img_gpu;
    uchar *y_gpu, *cb_gpu, *cr_gpu;
    uint *hist_gpu, *block_hist_gpu;

    if (4 != argc) {
        std::cerr << "Usage: ./main in_path out_path" << std::endl;
        return 1;
    }

    rgb_img = stbi_load(
        argv[1],
        &width,
        &height,
        &channels,
        3);
    if (NULL == rgb_img) {
        std::cerr << "Failed to load image" << std::endl;
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    
    if (0 == device_count) {
        std::cerr << "No CUDA device found" << std::endl;
        return 1;
    }

    int dev = 0;

    checkCudaErrors(cudaMalloc((void **)&rgb_img_gpu, width * height * 3));
    checkCudaErrors(cudaMalloc((void **)&y_gpu, width * height));
    checkCudaErrors(cudaMalloc((void **)&cb_gpu, width * height));
    checkCudaErrors(cudaMalloc((void **)&cr_gpu, width * height));
    checkCudaErrors(cudaMalloc((void **)&hist_gpu, 256 * sizeof(uint)));

    checkCudaErrors(cudaMemcpy(rgb_img_gpu, rgb_img, width * height * 3, cudaMemcpyHostToDevice));

    // dim3 threadsPerBlock(32, 32);
    // dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
    //                    (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(((width * height) - 1) / BLOCK_SIZE + 1);
	dim3 dimBlockHist(HIST_SIZE);
	dim3 dimGridHist(1);

    rgb_to_ycbcr<<<dimGrid, dimBlock>>>(rgb_img_gpu, y_gpu, cb_gpu, cr_gpu, width * height);

    cudaMemset(hist_gpu, 0, HIST_SIZE * sizeof(uint));

	dim3 dimBlockH(N_THREADS_PER_BLOCK);
	dim3 dimGridH(((width * height) - 1) / (N_THREADS_PER_BLOCK * 255) + 1);
	// dim3 dimGridH(((width * height) - 1) / N_THREADS_PER_BLOCK + 1);

    checkCudaErrors(cudaMalloc((void **)&block_hist_gpu, dimGridH.x * HIST_SIZE * sizeof(uint)));

    // histogram<<<dimGridH, dimBlockH>>>(y_gpu, hist_gpu, width * height);

    histogram<<<dimGridH, dimBlockH>>>(y_gpu, block_hist_gpu, width * height);

    merge_block_histograms<<<HIST_SIZE, MERGE_THREADBLOCK_SIZE>>>(hist_gpu, block_hist_gpu, dimGridH.x);

    uint hist[HIST_SIZE];

    cudaMemcpy(hist, hist_gpu, HIST_SIZE * sizeof(uint), cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaDeviceSynchronize());

    for (int i = 0; i < HIST_SIZE; ++i) {
        std::cout << hist[i] << std::endl;
    }

    checkCudaErrors(cudaDeviceSynchronize());

    float *cdf;
    checkCudaErrors(cudaMalloc((void **)&cdf, HIST_SIZE * sizeof(float)));

    calcCDF<<<dimGridHist, dimBlockHist>>>(cdf, hist_gpu, width * height);

    equalize<<<dimGrid, dimBlock>>>(y_gpu, cdf, width * height);

    ycbcr_to_rgb<<<dimGrid, dimBlock>>>(y_gpu, cb_gpu, cr_gpu, rgb_img_gpu, width * height);

    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    printf("Time to generate:  %f s \n", time / 1000);

    cudaMemcpy(rgb_img, rgb_img_gpu, width * height * 3, cudaMemcpyDeviceToHost);

    checkCudaErrors(cudaDeviceSynchronize());

    stbi_write_png(
        argv[2],
        width,
        height,
        3,
        rgb_img,
        width * 3);

    stbi_image_free(rgb_img);
    return 0;
}