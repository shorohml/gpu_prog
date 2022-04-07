#include "forward.h"
#include "img.h"
#include "App.h"
#include <fstream>
#include <iostream>
#include <memory>
#include <sstream>
#include <vector>

template <typename T>
T clamp(T val, T min_v, T max_v)
{
    if (val < min_v) {
        return min_v;
    } else if (val > max_v) {
        return max_v;
    }
    return val;
}

const std::string WHITESPACE = " \n\r\t\f\v";
 
std::string ltrim(const std::string &s)
{
    size_t start = s.find_first_not_of(WHITESPACE);
    return (start == std::string::npos) ? "" : s.substr(start);
}
 
std::string rtrim(const std::string &s)
{
    size_t end = s.find_last_not_of(WHITESPACE);
    return (end == std::string::npos) ? "" : s.substr(0, end + 1);
}
 
std::string trim(const std::string &s) {
    return rtrim(ltrim(s));
}

std::vector<float> read_weights(const std::string& path)
{
    std::ifstream in_file(path);
    std::vector<float> res;

    if (in_file.is_open()) {
        std::string line;
        while (std::getline(in_file, line)) {
            std::stringstream stream(line);

            std::string val;
            while (std::getline(stream, val, ',')) {
                if (0 == trim(val).size()) {
                    continue;
                }
                res.push_back(std::atof(val.c_str()));
            }
        }
    }
    return res;
}

int main(int argc, char** argv)
{
    if (argc != 3) {
        std::cerr << "usage: example-app <path-to-exported-script-module>\n";
        return -1;
    }

    int size = std::atoi(argv[2]);

    std::vector<std::vector<float> > weights;

    const int W = size;
    const int H = size;

    float* color;

    color = (float*)malloc(W * H * 3 * sizeof(float));

    std::vector<std::string> paths;
    for (int i = 0; i < 5; ++i) {
        paths.push_back(std::string("../data/NeuralNetworkWeights/weights" + std::to_string(i) + ".txt"));
        paths.push_back(std::string("../data/NeuralNetworkWeights/biases" + std::to_string(i) + ".txt"));
    }

    for (const auto& path: paths) {
        weights.push_back(
            read_weights(path)
        );
    }

    std::vector<int> sizes = {
        32 * 3,
        32,
        32 * 32,
        32,
        32 * 32,
        32,
        32 * 32,
        32,
        32,
        1
    };

    std::string pathToConfig = std::string("../config.json");
    App app(pathToConfig, weights, sizes);
    app.Run();

    forward(
        color,
        weights,
        W,
        H,
        1e-2);

    int idx = 0;
    Img img(W, H, 3);
    for (int i = 0; i < H; ++i) {
        for (int j = 0; j < W; ++j) {
            idx = (i * W + j) * 3;
            img.get_data()[idx] = 255 * clamp(color[idx], 0.0f, 1.0f);
            img.get_data()[idx + 1] = 255 * clamp(color[idx + 1], 0.0f, 1.0f);
            img.get_data()[idx + 2] = 255 * clamp(color[idx + 2], 0.0f, 1.0f);
        }
    }

    img.save("test.png");

    free(color);
    return 0;
}