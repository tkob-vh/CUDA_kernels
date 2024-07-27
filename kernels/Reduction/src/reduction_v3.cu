#include <reduction.hh>
/*
 * \file reduction_v3.cu
 * A reduction implementation without constraint of gridDim.x = 1.
 * blockDim.x = input.size / 2.
 * Without boundrary check now.
 * Tricks:  
 *  1. Reduce control divergence and improve the memory coalescing by rearrange 
 *     the assignment strategy.
 *  2. Use shared memory to minimize global memory accesses.
 *  3. Enable multi-gird kernel using atomicAdd.
 */

__global__ void reduction_v3(float *input, float *output) {
    extern __shared__ float shared_mem[];
    float *input_s = shared_mem;

    uint32_t segment = 2 * blockDim.x * blockIdx.x;
    uint32_t i_g = segment + threadIdx.x;
    input_s[threadIdx.x] = input[i_g] + input[i_g + blockDim.x];
    for(uint32_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        if(threadIdx.x < stride) {
            input_s[threadIdx.x] += input_s[threadIdx.x + stride];
        }
    }

    if(threadIdx.x == 0) {
        atomicAdd(output, input_s[0]);
    }
}