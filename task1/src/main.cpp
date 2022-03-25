#include "cpu/hist_eq.h"
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "cpu/hist_eq.h"
#include "gpu/hist_eq.h"
#include <cuda_runtime.h>
#include <omp.h>
#include <stb_image_write.h>
#include <string.h>

int main(int argc, char** argv)
{
    int width, height, channels;
    unsigned char *rgb_img, *rgb_img_copy, *ycbcr_img;
    float start, finish, time;
    cudaEvent_t start_cu, stop_cu;
    int device_count;

    if (4 != argc) {
        std::cerr << "Usage: ./main in_path out_path num_threads" << std::endl;
        return 1;
    }

    cudaGetDeviceCount(&device_count);

    if (0 == device_count) {
        std::cerr << "No CUDA device found" << std::endl;
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
    rgb_img_copy = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
    if (NULL == rgb_img_copy) {
        std::cerr << "Failed to alloc memory" << std::endl;
        stbi_image_free(rgb_img);
        return 1;
    }
    memcpy(rgb_img_copy, rgb_img, width * height * 3);

    omp_set_num_threads(std::atoi(argv[3]));

    start = omp_get_wtime();

    ycbcr_img = (unsigned char*)malloc(sizeof(unsigned char) * width * height * 3);
    if (NULL == ycbcr_img) {
        std::cerr << "Failed to alloc memory" << std::endl;
        stbi_image_free(rgb_img);
        stbi_image_free(rgb_img_copy);
        return 1;
    }
    equalize_rgb(
        rgb_img_copy,
        ycbcr_img,
        width,
        height);

    finish = omp_get_wtime();
    time = finish - start;

    std::cout << time * 1000 << std::endl;

    memcpy(rgb_img_copy, rgb_img, width * height * 3);

    cudaEventCreate(&start_cu);
    cudaEventCreate(&stop_cu);
    cudaEventRecord(start_cu, 0);

    equalize_histogram(
        rgb_img_copy,
        width,
        height);

    checkCudaErrors(cudaDeviceSynchronize());

    cudaEventRecord(stop_cu, 0);
    cudaEventSynchronize(stop_cu);
    cudaEventElapsedTime(&time, start_cu, stop_cu);

    std::cout << time << std::endl;

    stbi_write_png(
        argv[2],
        width,
        height,
        3,
        rgb_img_copy,
        width * 3);

    stbi_image_free(ycbcr_img);
    stbi_image_free(rgb_img);
    stbi_image_free(rgb_img_copy);
    return 0;
}