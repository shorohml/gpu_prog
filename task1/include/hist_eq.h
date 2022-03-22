#pragma once

void rgb_to_ycbcr(
    const unsigned char* rgb_img,
    unsigned char* ycbcr_img,
    const int width,
    const int height);

void ycbcr_to_rgb(
    const unsigned char* ycbcr_img,
    unsigned char* rgb_img,
    const int width,
    const int height);

void compute_hist(
    const unsigned char* ycbcr_img,
    unsigned int* hist,
    const int width,
    const int height);

void compute_cdf(
    const unsigned int* hist,
    double* cdf);

void equalize(
    unsigned char* ycbcr_img,
    const double* cdf,
    const int width,
    const int height);
