#include "cpu/hist_eq.h"
#include "gpu/hist_eq.h"
#include "img.h"
#include <iostream>

const int num_threads = 8;

int main(int argc, char** argv)
{
    if (3 != argc) {
        std::cerr << "Usage: ./main input_image_path n_runs" << std::endl;
        return 1;
    }

    int device_count;
    cudaGetDeviceCount(&device_count);
    if (0 == device_count) {
        std::cerr << "No CUDA device found" << std::endl;
        return 1;
    }

    const int n_runs = std::atoi(argv[2]);
    double time = 0.0;
    double time_w_copy = 0.0;
    double time_wo_copy = 0.0;

    Img rgb_img(argv[1], 3);
    Img rgb_img_copy_1(rgb_img);
    Img rgb_img_copy_2(rgb_img);

    EqualizeHistogramCPU filter_cpu(1);
    for (int i = 0; i < n_runs; ++i) {
        rgb_img_copy_1 = rgb_img;
        filter_cpu.process(rgb_img_copy_1);
        time += filter_cpu.get_time();
    }
    std::cout << "Filter execution time (in ms):" << std::endl;
    std::cout << "CPU time (1 thread): " << time / n_runs << std::endl;

    filter_cpu.set_num_thread(num_threads);
    time = 0.0;
    for (int i = 0; i < n_runs; ++i) {
        rgb_img_copy_1 = rgb_img;
        filter_cpu.process(rgb_img_copy_1);
        time += filter_cpu.get_time();
    }
    std::cout << "CPU time (" << num_threads << " threads): " << time / n_runs << std::endl;

    EqualizeHistogramGPU filter_gpu(HistComputationMode::DEFAULT);
    for (int i = 0; i < n_runs; ++i) {
        rgb_img_copy_2 = rgb_img;
        filter_gpu.process(rgb_img_copy_2);
        time_w_copy += filter_gpu.get_time_w_copy();
        time_wo_copy += filter_gpu.get_time_wo_copy();
    }
    std::cout << "GPU time without copy: " << time_wo_copy / n_runs << std::endl;
    std::cout << "GPU time with copy: " << time_w_copy / n_runs << std::endl;
    std::cout << "GPU copy time: " << (time_w_copy - time_wo_copy) / n_runs << std::endl;

    rgb_img_copy_1.save("out_cpu.png");
    rgb_img_copy_2.save("out_gpu.png");
    return 0;
}