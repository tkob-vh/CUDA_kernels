#include <cuda_runtime.h>
#include <cstdint>
#include <cassert>


__global__ void reduction_v0(float *input, float *output);
void reduction_v0_invok(float *input, float *output, uint64_t num);

__global__ void reduction_v1(float *input, float *output);


__global__ void reduction_v2(float *input, float *output);


__global__ void reduction_v3(float *input, float *output);



#define COARSE_FACTOR 4
__global__ void reduction_v4(float *input, float *output);