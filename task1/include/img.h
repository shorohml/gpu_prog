#pragma once

#include <cassert>
#include <memory>
#include <stb_image.h>
#include <stb_image_write.h>
#include <string>

#define uchar unsigned char

struct Img {
private:
    std::shared_ptr<uchar> _data;
    int _width = 0;
    int _height = 0;
    int _channels = 0;

public:
    Img(const std::string& path, int force_channels = 0)
    {
        load(path, force_channels);
    }

    Img(const int width, const int height, const int channels);

    Img(const Img& img);

    void from(const Img& img);

    Img& operator=(const Img& img);

    void load(const std::string& path, int force_channels = 0);

    void save(const std::string& path);

    uchar* get_data() const
    {
        return _data.get();
    }

    int get_width() const
    {
        return _width;
    }

    int get_height() const
    {
        return _height;
    }

    int get_channels() const
    {
        return _channels;
    }
};
