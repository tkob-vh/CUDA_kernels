#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#define BLOCK_WIDTH 16


__global__ void convolution_v0(float *N, float *F, float *P, int r, int width, int height);

#define FILTER_RADIUS 3
__constant__ float F[FILTER_RADIUS*2+1][FILTER_RADIUS*2+1];
__global__ void convolution_v1(float *N, float *P, int r, int width, int height);

__global__ void convolution_v2(float *N, float *P, int r, int width, int height);

