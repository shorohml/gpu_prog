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


__device__ float clamp(float x, float a, float b)
{
    return max(a, min(b, x));
}


__global__ void rgb_to_ycbcr(uchar3 *rgb, uchar3 *ycbcr, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    uint i = row * width + col;

    unsigned char r = rgb[i].x;
    unsigned char g = rgb[i].y;
    unsigned char b = rgb[i].z;

    ycbcr[i].x = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
    ycbcr[i].y = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
    ycbcr[i].z = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;
}


__global__ void ycbcr_to_rgb(uchar3 *ycbcr, uchar3 *rgb, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    uint i = row * width + col;

    float y = (float)ycbcr[i].x - 16.;
    float cb = (float)ycbcr[i].y - 128.;
    float cr = (float)ycbcr[i].z - 128.;

    float r = 1.164 * y + 1.596 * cr;
    float g = 1.164 * y - 0.392 * cb - 0.813 * cr;
    float b = 1.164 * y + 2.017 * cb;

    rgb[i].x = clamp(r, 0., 255.);
    rgb[i].y = clamp(g, 0., 255.);
    rgb[i].z = clamp(b, 0., 255.);
}


// __global__ void histogram(uchar3 *ycbcr, uint *hist, int width, int height)
// { 
//     int row = blockIdx.y * blockDim.y + threadIdx.y;
//     int col = blockIdx.x * blockDim.x + threadIdx.x;

//     if (row >= height || col >= width) {
//         return;
//     }

//     uint i = row * width + col;
//     unsigned char y = ycbcr[i].x;

//     __shared__ uint shared_hist[32][256];
//     for (int i = 0; i < 256 / 32; ++i) {
//         shared_hist[threadIdx.x][threadIdx.y + i * 32] = 0;
//     }

//     __syncthreads();

//     uint *warp_hist = shared_hist[threadIdx.x];
//     atomicAdd(warp_hist + y, 1);

//     __syncthreads();

//     uint idx = threadIdx.x * 32 + threadIdx.y;
//     if (idx >= 256) {
//         return;
//     }

//     uint sum = 0;
//     for (int j = 0; j < 32; ++j) {
//         sum += shared_hist[j][idx];
//     }
//     atomicAdd(hist + idx, sum);
// }


__global__ void histogram(uchar3 *ycbcr, uint *hist, int width, int height)
{ 
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    uint i = row * width + col;
    unsigned char y = ycbcr[i].x;

    atomicAdd(hist + y, 1);
}


#define N_THREADS_PER_BLOCK 1920
#define HIST_SIZE 256
#define SHARED_S N_THREADS_PER_BLOCK * HIST_SIZE


// __global__ void histogram(uchar3 *ycbcr, uint *hist, int size)
// {
// 	uint i = blockDim.x * blockIdx.x + threadIdx.x;

//     if (i >= size) {
//         return;
//     }

//     unsigned char y = ycbcr[i].x;

//     __shared__ uint shared_hist[SHARED_S];

//     printf("%d\n", SHARED_S);

//     shared_hist[0] = 0;
//     ycbcr[i].x = shared_hist[0];
// //     __syncthreads();

// //     uint idx = threadIdx.x * 32 + threadIdx.y;
// //     if (idx >= 256) {
// //         return;
// //     }

// //     uint sum = 0;
// //     for (int j = 0; j < 32; ++j) {
// //         sum += shared_hist[j][idx];
// //     }
// //     atomicAdd(hist + idx, sum);
// }


__global__ void calcCDF(
        float *cdf,
        uint *hist,
	    int width,
        int height) {
	__shared__ float cdf_shared[256];
	int i = blockDim.x * blockIdx.x + threadIdx.x;

	if (i < 256) {
        cdf_shared[i] = (float)hist[i] / (float)(width * height);
    }

    __syncthreads();

	for (uint stride = 1; stride <= BLOCK_SIZE; stride *= 2) {
		uint index = (threadIdx.x + 1) * stride * 2 - 1;
		if (index < 256)
			cdf_shared[index] += cdf_shared[index - stride];
		__syncthreads();
	}

	for (uint stride = BLOCK_SIZE / 2; stride>0; stride /= 2) {
		__syncthreads();
		uint index = (threadIdx.x + 1)*stride * 2 - 1;
		if (index + stride < SCAN_SIZE && index + stride < 256) {
			cdf_shared[index + stride] += cdf_shared[index];
		}
	}

	__syncthreads();

	if (i < 256) {
		cdf[i] = cdf_shared[threadIdx.x];
    }
}


__global__ void equalize(uchar3 *ycbcr, float *cdf, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row >= height || col >= width) {
        return;
    }

    uint i = row * width + col;

    unsigned char y = ycbcr[i].x;
    ycbcr[i].x = 219 * cdf[y] + 16;
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
    unsigned char *rgb_img, *ycbcr_img;
    uchar3 *rgb_img_gpu, *ycbcr_img_gpu;
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

    ycbcr_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
    if (NULL == ycbcr_img) {
        std::cerr << "Failed to alloc memory for YCbCr image" << std::endl;
        stbi_image_free(rgb_img);
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
    checkCudaErrors(cudaMalloc((void **)&ycbcr_img_gpu, width * height * 3));
    checkCudaErrors(cudaMalloc((void **)&hist_gpu, 256 * sizeof(uint)));

    checkCudaErrors(cudaMemcpy(rgb_img_gpu, rgb_img, width * height * 3, cudaMemcpyHostToDevice));

    dim3 threadsPerBlock(32, 32);
    dim3 blocksPerGrid((width + threadsPerBlock.x - 1) / threadsPerBlock.x,
                       (height + threadsPerBlock.y - 1) / threadsPerBlock.y);

    checkCudaErrors(cudaMalloc((void **)&block_hist_gpu, blocksPerGrid.x * blocksPerGrid.y * 256 * sizeof(uint)));


    float time;
    cudaEvent_t start, stop;

    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start, 0);

    rgb_to_ycbcr<<<blocksPerGrid, threadsPerBlock>>>(rgb_img_gpu, ycbcr_img_gpu, width, height);

    cudaMemset(hist_gpu, 0, 256 * sizeof(uint));

	// dim3 dimBlockH(N_THREADS_PER_BLOCK);
	// dim3 dimGridH(((width * height) - 1) / (N_THREADS_PER_BLOCK * 255) + 1);
    // histogram<<<dimGridH, dimBlockH>>>(ycbcr_img_gpu, hist_gpu, width * height);

    checkCudaErrors(cudaDeviceSynchronize());

    histogram<<<blocksPerGrid, threadsPerBlock>>>(ycbcr_img_gpu, hist_gpu, width, height);

	dim3 dimBlock(BLOCK_SIZE);
	dim3 dimGrid(((width*height) - 1) / BLOCK_SIZE + 1);
	dim3 dimGridHist(((width*height) - 1) / HISTOGRAM_LENGTH + 1);

    float *cdf;
    checkCudaErrors(cudaMalloc((void **)&cdf, 256 * sizeof(float)));

    calcCDF<<<dimGridHist, dimBlock>>>(cdf, hist_gpu, width, height);

    equalize<<<blocksPerGrid, threadsPerBlock>>>(ycbcr_img_gpu, cdf, width, height);

    ycbcr_to_rgb<<<blocksPerGrid, threadsPerBlock>>>(ycbcr_img_gpu, rgb_img_gpu, width, height);


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

    stbi_image_free(ycbcr_img);
    stbi_image_free(rgb_img);
    return 0;
}