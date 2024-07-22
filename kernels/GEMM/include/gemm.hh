#include <cuda_runtime.h>
#include <cstdint>

#define TILE_WIDTH 16

__global__ void matrixMul0(float *M, float *N, float *P, int Width);
__global__ void matrixMul1(float *M, float *N, float *P, int r, int s, int t);
__global__ void matrixMul2(float *M, float *N, float *P, int r, int s, int t, unsigned int Mds_sz, unsigned int Nds_sz);
__global__ void matrixMul3(float *M, float *N, float *P, int r, int s, int t, unsigned int Mds_sz, unsigned int Nds_sz);