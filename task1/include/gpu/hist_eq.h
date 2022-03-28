#pragma once

#include "img.h"
#include <cstdio>
#include <cuda_runtime.h>
#include <stdexcept>

#define N_THREADS_PER_BLOCK 192
#define HIST_SIZE 256
#define SHARED_S N_THREADS_PER_BLOCK* HIST_SIZE
#define MERGE_THREADBLOCK_SIZE 256
#define uchar unsigned char

template <typename T>
inline void check(T result, char const* const func, const char* const file,
    int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<uint>(result), cudaGetErrorName(result), func);
        throw std::runtime_error("");
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

struct HistData {
public:
    uint* hist_gpu;
    uint* block_hist_gpu;
    uchar3* rgb_gpu;
    uchar* y_gpu;
    uchar* cb_gpu;
    uchar* cr_gpu;
    float* cdf_gpu;

    HistData(const int width, const int height, const int channels);

    ~HistData();
};

enum HistComputationMode {
    DEFAULT,
    SEPARATE_THREAD_HIST,
};

class EqualizeHistogramGPU {
private:
    float _time_w_copy;
    float _time_wo_copy;
    HistComputationMode _mode;

public:
    EqualizeHistogramGPU(HistComputationMode hist_mode)
    {
        _mode = hist_mode;
    }

    void process(Img& rgb_img);

    float get_time_w_copy() const
    {
        return _time_w_copy;
    }

    float get_time_wo_copy() const
    {
        return _time_wo_copy;
    }
};
