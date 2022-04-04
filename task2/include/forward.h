#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

#define N_THREADS_PER_BLOCK 1024

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

void forward(
    float* in,
    float* out,
    float** weights,
    int W,
    int H);

void sphere_tracing(
    float* P,
    float* D,
    float eps,
    float **weights,
    int W,
    int H);
