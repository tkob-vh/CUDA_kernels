#include <cuda_runtime.h>

const float c0 = 1 / 3, c1 = 1 / 6, c2 = 1 / 6, c3 = 1 / 6, c4 = 1 / 6, c5 = 1 / 6, c6 = 1 / 6;

__global__ void stencil_v0(const float *in, float *out, int nx, int ny, int nz);

__global__ void stencil_v1(const float *in, float *out, int nx, int ny, int nz);