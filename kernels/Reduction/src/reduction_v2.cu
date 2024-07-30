#include <reduction.hh>
/*
 * \file reduction_v2.cu
 * A reduction implementation with gridDim.x = 1
 * blockDim.x = input.size / 2.
 * Without boundrary check now.
 * Tricks:  
 *  1. Reduce control divergence and improve the memory coalescing by rearrange 
 *     the assignment strategy.
 *  2. Use shared memory to minimize global memory accesses
 */

__global__ void reduction_v2(float *input, float *output) {
    extern __shared__ float shared_mem[];
    float *input_s = shared_mem;

    uint32_t i = threadIdx.x;
    input_s[i] = input[i] + input[i + blockDim.x];
    for(uint32_t stride = blockDim.x / 2; stride >= 1; stride /= 2) {
        __syncthreads();
        input_s[i] += input_s[i + stride];
    }

    if(threadIdx.x == 0) {
        *output = input_s[0];
    }
}

void reduction_v2_invok(float *input, float *output, uint64_t num) {
    assert(num <= 2048 && num % 2 == 0);
    dim3 blockDim(num / 2, 1, 1);
    dim3 gridDim(1, 1, 1);

    reduction_v2<<<gridDim, blockDim>>>(input, output);
}