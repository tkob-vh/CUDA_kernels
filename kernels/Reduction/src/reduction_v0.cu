#include <reduction.hh>
/*
 * \file reduction_v0.cu
 * A naive reduction implementation with gridDim.x = 1
 * blockDim.x = input.size / 2.
 * Without boundrary check now.
*/

__global__ void reduction_v0(float *input, float *output) {
    uint32_t i = 2 * threadIdx.x;
    for(uint32_t stride = 1; stride <= blockDim.x; stride *= 2) {
        if(threadIdx.x % stride == 0) {
            input[i] += input[i + stride];
        }
        __syncthreads();
    }
    if(threadIdx.x == 0) {
        *output = input[0];
    }
}

void reduction_v0_invok(float *input, float *output, uint64_t num) {
    assert(num <= 2048 && num % 2 == 0);
    dim3 blockDim(num / 2, 1, 1);
    dim3 gridDim(1, 1, 1);

    reduction_v0<<<gridDim, blockDim>>>(input, output);
}