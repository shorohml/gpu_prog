#pragma once

#include <cuda_runtime.h>
#include <stdexcept>

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
