#include "hist_eq.h"
#include <fstream>
#include <iostream>
#define STB_IMAGE_IMPLEMENTATION
#include <stb_image.h>
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <omp.h>
#include <stb_image_write.h>

int main(int argc, char** argv)
{
    int width, height, channels;
    unsigned char *rgb_img, *ycbcr_img;
    unsigned int hist[256];
    double cdf[256];

    if (4 != argc) {
        std::cerr << "Usage: ./main in_path out_path" << std::endl;
        return 1;
    }

    rgb_img = stbi_load(
        argv[1],
        &width,
        &height,
        &channels,
        STBI_rgb);
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

    omp_set_num_threads(std::atoi(argv[3]));

    double start = omp_get_wtime();

    rgb_to_ycbcr(
        rgb_img,
        ycbcr_img,
        width,
        height);
    compute_hist(
        ycbcr_img,
        hist,
        width,
        height);
    compute_cdf(
        hist,
        cdf);
    equalize(
        ycbcr_img,
        cdf,
        width,
        height);
    ycbcr_to_rgb(
        ycbcr_img,
        rgb_img,
        width,
        height);

    double end = omp_get_wtime();

    std::cout << end - start << std::endl;

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