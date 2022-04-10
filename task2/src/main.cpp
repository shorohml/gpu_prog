#include "ray_marching/nn_weights.h"
#include "img.h"
#include "App.h"
#include <fstream>

int main(int argc, char **argv)
{
    int device_count;
    cudaGetDeviceCount(&device_count);
    if (0 == device_count) {
        std::cerr << "No CUDA device found" << std::endl;
        return 1;
    }

    nlohmann::json config;
    std::vector<std::vector<float> > weights;

    std::string config_path("../config.json");
    if (2 == argc) {
        config_path = std::string(argv[1]);
    }

    std::ifstream input(config_path);
    if (!input.good()) {
        std::cerr << "Failed to load config" << std::endl;
        return 1;
    }
    input >> config;

    std::vector<std::string> paths;
    std::string data_path(config["dataPath"]); 
    for (int i = 0; i < 5; ++i) {
        paths.push_back(data_path + std::string("/NeuralNetworkWeights/weights") + std::to_string(i) + ".txt");
        paths.push_back(data_path + std::string("/NeuralNetworkWeights/biases") + std::to_string(i) + ".txt");
    }

    for (const auto& path: paths) {
        weights.push_back(
            read_weights(path)
        );
    }

    std::string pathToConfig = std::string("../config.json");
    App app(config, weights);
    return app.Run();
}