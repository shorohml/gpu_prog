__global__ void rgb_to_ycbcr(uchar3 *rgb, uchar3 *ycbcr, int width, int height)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row > height || col > width) {
        return;
    }

    unsigned int i = row * width + col;

    unsigned char r = rgb[i].x;
    unsigned char g = rgb[i].y;
    unsigned char b = rgb[i].z;

    ycbcr[i].x = 0.257 * r + 0.504 * g + 0.098 * b + 16.0;
    ycbcr[i].y = -0.148 * r - 0.291 * g + 0.439 * b + 128.0;
    ycbcr[i].z = 0.439 * r - 0.368 * g - 0.071 * b + 128.0;
}