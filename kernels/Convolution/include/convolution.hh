#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <cstdint>

#define BLOCK_WIDTH 16


__global__ void convolution_v0(float *N, float *F, float *P,
                                int r, int width, int height);

void convolution_v0_invok(uint32_t width, uint32_t height, uint32_t r,
                            float *in, float *out, float *filter);

#define FILTER_RADIUS 3
__constant__ float F[FILTER_RADIUS*2+1][FILTER_RADIUS*2+1];
__global__ void convolution_v1(float *N, float *P, int width, int height);

void convolution_v1_invok(uint32_t width, uint32_t height,
                            float *in, float *out);


#define IN_TILE_WIDTH BLOCK_WIDTH
#define OUT_TILE_WIDTH (IN_TILE_WIDTH-2*FILTER_RADIUS)
__global__ void convolution_v2(float *N, float *P, int width, int height);

void convolution_v2_invok(uint32_t width, uint32_t height,
                            float *in, float *out);
