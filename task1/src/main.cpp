#include "cpu/hist_eq.h"
#include "gpu/hist_eq.h"
#include "img.h"
#include <iostream>

int main(int argc, char** argv)
{
    if (4 != argc) {
        std::cerr << "Usage: ./main in_path out_path_cpu out_path_gpu" << std::endl;
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (0 == device_count) {
        std::cerr << "No CUDA device found" << std::endl;
        return 1;
    }

    Img rgb_img(argv[1], 3);
    Img rgb_img_copy_1(rgb_img);
    Img rgb_img_copy_2(rgb_img);

    EqualizeHistogramCPU filter_cpu(1);
    filter_cpu.process(rgb_img);
    std::cout << "Filter execution time (in ms):" << std::endl;
    std::cout << "CPU time (1 thread): " << filter_cpu.get_time() << std::endl;

    filter_cpu.set_num_thread(8);
    filter_cpu.process(rgb_img_copy_1);
    std::cout << "CPU time (8 threads): " << filter_cpu.get_time() << std::endl;

    EqualizeHistogramGPU filter_gpu;
    filter_gpu.process(rgb_img_copy_2);
    std::cout << "GPU time without copy: " << filter_gpu.get_time_wo_copy() << std::endl;
    std::cout << "GPU time with copy: " << filter_gpu.get_time_w_copy() << std::endl;
    std::cout << "GPU copy time: " << filter_gpu.get_time_w_copy() - filter_gpu.get_time_wo_copy() << std::endl;

    rgb_img_copy_1.save(argv[2]);
    rgb_img_copy_2.save(argv[3]);
    return 0;
}