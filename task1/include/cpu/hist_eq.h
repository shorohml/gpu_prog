#pragma once

#include "img.h"
#include <omp.h>

void rgb_to_ycbcr(
    const unsigned char* rgb_img,
    unsigned char* ycbcr_img,
    const int width,
    const int height);

void ycbcr_to_rgb(
    const unsigned char* ycbcr_img,
    unsigned char* rgb_img,
    const int width,
    const int height);

void compute_hist(
    const unsigned char* ycbcr_img,
    unsigned int* hist,
    const int width,
    const int height);

void compute_cdf(
    const unsigned int* hist,
    double* cdf);

void equalize_ycbcr(
    unsigned char* ycbcr_img,
    const double* cdf,
    const int width,
    const int height);

class EqualizeHistogramCPU {
private:
    int _num_threads;
    double _time = 0.;

public:
    EqualizeHistogramCPU(int num_threads)
    {
        set_num_thread(num_threads);
    }

    ~EqualizeHistogramCPU() { }

    void set_num_thread(int num_threads)
    {
        omp_set_num_threads(num_threads);
        _num_threads = num_threads;
    }

    int get_num_threads() const
    {
        return _num_threads;
    }

    double get_time() const
    {
        return _time;
    }

    void process(Img& rgb_img);
};
