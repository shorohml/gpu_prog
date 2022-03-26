#ifndef STB_IMAGE_IMPLEMENTATION
#define STB_IMAGE_IMPLEMENTATION
#endif
#ifndef STB_IMAGE_WRITE_IMPLEMENTATION
#define STB_IMAGE_WRITE_IMPLEMENTATION
#endif
#include "img.h"
#include <string>

Img::Img(const int width, const int height, const int channels)
{
    _width = width;
    _height = height;
    _channels = channels;
    _data = std::shared_ptr<uchar>(
        (uchar*)malloc(width * height * channels),
        stbi_image_free);
}

void Img::load(const std::string& path, int force_channels)
{
    uchar* img = stbi_load(path.c_str(), &_width, &_height, &_channels, force_channels);
    if (img == nullptr) {
        throw std::runtime_error("Couldn't load image " + path);
    }
    _data = std::shared_ptr<uchar>(
        img,
        stbi_image_free);
    if (force_channels) {
        _channels = force_channels;
    }
}

Img::Img(const Img& img)
{
    _width = img.get_width();
    _height = img.get_height();
    _channels = img.get_channels();
    int size = _width * _height * _channels;
    _data = std::shared_ptr<uchar>(
        (uchar*)malloc(size),
        stbi_image_free);
    memcpy(_data.get(), img.get_data(), size);
}

void Img::save(const std::string& path)
{
    stbi_write_png(
        path.c_str(),
        _width,
        _height,
        _channels,
        _data.get(),
        _width * 3);
}