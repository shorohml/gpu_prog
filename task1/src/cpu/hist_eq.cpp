#include "cpu/hist_eq.h"
#include <algorithm>
#include <iostream>
#include <omp.h>

namespace {
template <typename T>
T clamp(const T& val, const T& l, const T& u)
{
    if (val < l) {
        return l;
    }
    if (val > u) {
        return u;
    }
    return val;
}
}

void rgb_to_ycbcr(
    const unsigned char* rgb_img,
    unsigned char* ycbcr_img,
    const int width,
    const int height)
{
    int i0, i1, i2;

#pragma omp parallel for private(i0, i1, i2) collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            i0 = (i * width + j) * 3;
            i1 = i0 + 1;
            i2 = i0 + 2;
            ycbcr_img[i0] = 0.257 * rgb_img[i0] + 0.504 * rgb_img[i1] + 0.098 * rgb_img[i2] + 16.0;
            ycbcr_img[i1] = -0.148 * rgb_img[i0] - 0.291 * rgb_img[i1] + 0.439 * rgb_img[i2] + 128.0;
            ycbcr_img[i2] = 0.439 * rgb_img[i0] - 0.368 * rgb_img[i1] - 0.071 * rgb_img[i2] + 128.0;
        }
    }
}

void ycbcr_to_rgb(
    const unsigned char* ycbcr_img,
    unsigned char* rgb_img,
    const int width,
    const int height)
{
    int i0, i1, i2;
    int y, cb, cr;

#pragma omp parallel for private(i0, i1, i2, y, cb, cr) collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            i0 = (i * width + j) * 3;
            i1 = i0 + 1;
            i2 = i0 + 2;
            y = static_cast<int>(ycbcr_img[i0]) - 16;
            cb = static_cast<int>(ycbcr_img[i1]) - 128;
            cr = static_cast<int>(ycbcr_img[i2]) - 128;
            rgb_img[i0] = clamp(1.164 * y + 1.596 * cr, 0.0, 255.0);
            rgb_img[i1] = clamp(1.164 * y - 0.392 * cb - 0.813 * cr, 0.0, 255.0);
            rgb_img[i2] = clamp(1.164 * y + 2.017 * cb, 0.0, 255.0);
        }
    }
}

void compute_hist(
    const unsigned char* ycbcr_img,
    unsigned int* hist,
    const int width,
    const int height)
{
    for (int i = 0; i < 256; ++i) {
        hist[i] = 0;
    }

#pragma omp parallel
    {
        int i0;
        unsigned int local_hist[256];

        for (int i = 0; i < 256; ++i) {
            local_hist[i] = 0;
        }

#pragma omp for nowait
        for (int i = 0; i < height; ++i) {
            for (int j = 0; j < width; ++j) {
                i0 = (i * width + j) * 3;
                ++local_hist[ycbcr_img[i0]];
            }
        }

#pragma omp critical
        {
            for (int i = 0; i < 256; ++i) {
                hist[i] += local_hist[i];
            }
        }
    }
}

void compute_cdf(
    const unsigned int* hist,
    double* cdf)
{
    for (int i = 0; i < 256; ++i) {
        cdf[i] = 0;
    }
    for (int i = 0; i < 256; ++i) {
        for (int j = i; j < 256; ++j) {
            cdf[j] += hist[i];
        }
    }
    for (int i = 0; i < 256; ++i) {
        cdf[i] /= cdf[255];
    }
}

void equalize_ycbcr(
    unsigned char* ycbcr_img,
    const double* cdf,
    const int width,
    const int height)
{
    int i0;

#pragma omp parallel for private(i0) collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            i0 = (i * width + j) * 3;
            ycbcr_img[i0] = 219 * cdf[ycbcr_img[i0]] + 16;
        }
    }
}

void equalize_rgb(
    unsigned char* rgb_img,
    unsigned char* ycbcr_img,
    const int width,
    const int height)
{
    unsigned int hist[256];
    double cdf[256];

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
    equalize_ycbcr(
        ycbcr_img,
        cdf,
        width,
        height);
    ycbcr_to_rgb(
        ycbcr_img,
        rgb_img,
        width,
        height);
}
