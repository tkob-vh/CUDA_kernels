#include <cuda_runtime.h>

const float c0 = 1 / 3, c1 = 1 / 6, c2 = 1 / 6, c3 = 1 / 6, c4 = 1 / 6, c5 = 1 / 6, c6 = 1 / 6;

__global__ void stencil_v0(const float *in, float *out, int nx, int ny, int nz);



// The IN_TILE_WIDTH should be the same as the block width.
#define IN_TILE_WIDTH 8
#define OUT_TILE_WIDTH 6
__global__ void stencil_v1(const float *in, float *out, int nx, int ny, int nz);


#define IN_TILE_WIDTH2 32
#define OUT_TILE_WIDTH2 30
__global__ void stencil_v2(const float *in, float *out, int nx, int ny, int nz);

__global__ void stencil_v3(const float *in, float *out, int nx, int ny, int nz);