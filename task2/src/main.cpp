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

int main()
{
    std::vector<std::vector<float> > weights;

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

    std::string pathToConfig = std::string("../config.json");
    App app(pathToConfig, weights);
    app.Run();

    return 0;
}