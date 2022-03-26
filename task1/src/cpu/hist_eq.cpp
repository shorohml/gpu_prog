#include "cpu/hist_eq.h"
#include <algorithm>
#include <chrono>
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
    const uchar* rgb_img,
    uchar* ycbcr_img,
    const int width,
    const int height)
{
    int i0, i1, i2, r, g ,b;

#pragma omp parallel for private(i0, i1, i2, r, g, b) collapse(2)
    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            i0 = (i * width + j) * 3;
            i1 = i0 + 1;
            i2 = i0 + 2;

            r = rgb_img[i0];
            g = rgb_img[i1];
            b = rgb_img[i2];

            ycbcr_img[i0] = (257 * r + 504 * g + 98 * b) / 1000 + 16;
            ycbcr_img[i1] = (-148 * r - 291 * g + 439 * b) / 1000 + 128;
            ycbcr_img[i2] = (439 * r - 368 * g - 71 * b) / 1000 + 128;
        }
    }
}

void ycbcr_to_rgb(
    const uchar* ycbcr_img,
    uchar* rgb_img,
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
            rgb_img[i0] = clamp((1164 * y + 1596 * cr) / 1000, 0, 255);
            rgb_img[i1] = clamp((1164 * y - 392 * cb - 813 * cr) / 1000, 0, 255);
            rgb_img[i2] = clamp((1164 * y + 2017 * cb) / 1000, 0, 255);
        }
    }
}

void compute_hist(
    const uchar* ycbcr_img,
    uint* hist,
    const int width,
    const int height)
{
    for (int i = 0; i < 256; ++i) {
        hist[i] = 0;
    }

#pragma omp parallel
    {
        int i0;
        uint local_hist[256];

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
    const uint* hist,
    float* cdf)
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
    uchar* ycbcr_img,
    const float* cdf,
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

void EqualizeHistogramCPU::process(Img& rgb_img)
{
    uint hist[256];
    float cdf[256];
    const int width = rgb_img.get_width();
    const int height = rgb_img.get_height();
    const int channels = rgb_img.get_channels();
    Img ycbcr_img(
        width,
        height,
        channels);

    auto t1 = std::chrono::high_resolution_clock::now();

    rgb_to_ycbcr(
        rgb_img.get_data(),
        ycbcr_img.get_data(),
        width,
        height);
    compute_hist(
        ycbcr_img.get_data(),
        hist,
        width,
        height);
    compute_cdf(
        hist,
        cdf);
    equalize_ycbcr(
        ycbcr_img.get_data(),
        cdf,
        width,
        height);
    ycbcr_to_rgb(
        ycbcr_img.get_data(),
        rgb_img.get_data(),
        width,
        height);

    auto t2 = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> ms_double = t2 - t1;
    _time = ms_double.count();
}
