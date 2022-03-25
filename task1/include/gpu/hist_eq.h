#pragma once

#include <cstdio>
#include <cuda_runtime.h>

#define N_THREADS_PER_BLOCK 192
#define HIST_SIZE 256
#define SHARED_S N_THREADS_PER_BLOCK* HIST_SIZE
#define MERGE_THREADBLOCK_SIZE 256
#define uchar unsigned char

template <typename T>
void check(T result, char const* const func, const char* const file,
    int const line)
{
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line,
            static_cast<uint>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

void equalize_histogram(
    uchar* rgb_cpu,
    int width,
    int height);
