#include "forward.h"
#include "img.h"
#include <iostream>
#include <memory>
#include <torch/script.h>

int main(int argc, char** argv)
{
    if (argc != 2) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    torch::jit::script::Module module;
    try {
        module = torch::jit::load(argv[1]);
    } catch (const c10::Error& e) {
        std::cerr << "error loading the model\n";
        return -1;
    }

    float** weights;
    weights = (float**)malloc(module.named_parameters().size() * sizeof(float*));

    std::int64_t idx = 0;
    for (auto named_param : module.named_parameters()) {
        weights[idx++] = named_param.value.data_ptr<float>();
    }

    const int W = 1024;
    const int H = 1024;

    float* points;
    float* out;

    points = (float*)malloc(W * H * 3 * sizeof(float));
    out = (float*)malloc(W * H * sizeof(float));

    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            idx = (i * W + j) * 3;
            points[idx + 2] = 1.0;
            points[idx + 1] = -(static_cast<float>(i) / H - 0.5) * 2;
            points[idx] = (static_cast<float>(j) / W - 0.5) * 2;
        }
    }

    forward(
        points,
        out,
        weights,
        W,
        H);

    Img img(W, H, 3);
    int idx_out;
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            idx = (i * W + j) * 3;
            idx_out = (i * W + j);
            if (out[idx_out] > 0.5) {
                img.get_data()[idx] = 255;
                img.get_data()[idx + 1] = 255;
                img.get_data()[idx + 2] = 255;
            } else {
                img.get_data()[idx] = 255;
                img.get_data()[idx + 1] = 0;
                img.get_data()[idx + 2] = 0;            
            }
        }
    }

    img.save("test.png");

    free(weights);
    free(points);
    free(out);
    return 0;
}