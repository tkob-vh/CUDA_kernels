#include <cuda_runtime.h>
#include <cstdint>

#define TILE_WIDTH 16

__global__ void gemm_v0(float *M, float *N, float *P, int Width);
__global__ void gemm_v1(float *M, float *N, float *P, int r, int s, int t);
__global__ void gemm_v2(float *M, float *N, float *P, int r, int s, int t, 
                        unsigned int Mds_sz, unsigned int Nds_sz);
__global__ void gemm_v3(float *M, float *N, float *P, int r, int s, int t, 
                        unsigned int Mds_sz, unsigned int Nds_sz);

void gemm_v0_invok(uint32_t n1, uint32_t n2, uint32_t n3,
                    float *a, float *b, float *c);
void gemm_v1_invok(uint32_t n1, uint32_t n2, uint32_t n3,
                    float *a, float *b, float *c);
void gemm_v2_invok(uint32_t n1, uint32_t n2, uint32_t n3,
                    float *a, float *b, float *c);
void gemm_v3_invok(uint32_t n1, uint32_t n2, uint32_t n3,
                    float *a, float *b, float *c);

