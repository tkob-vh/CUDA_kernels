#include <reduction.hh>
/*
 * \file reduction_v1.cu
 * A reduction implementation with gridDim.x = 1
 * blockDim.x = input.size / 2.
 * Without boundrary check now.
 * Tricks: 
 *  1. Reduce control divergence and improve the memory coalescing by rearrange 
 *     the assignment strategy.
 */


__global__ void reduction_v1(float *input, float *output) {
    uint32_t i = threadIdx.x;
    for(uint32_t stride = blockDim.x; stride >=1; stride /= 2) {
        if(threadIdx.x < stride) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }

}