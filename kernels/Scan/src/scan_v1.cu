#include "scan.hh"

/*
 * \file scan_v1.cu
 * \Description: A parallel scan using Brent-Kung algorithm.
*/

__global__ void scan_v1(float *input, float *output, uint32_t N) {
    __shared__ float intermediate[SECTION_SIZE];
    uint32_t tx = threadIdx.x;
    uint32_t i = 2 * blockDim.x * blockIdx.x + tx;
    if(i < N) {
        intermediate[tx] = input[i];
    }
    if(i + blockDim.x < N) {
        intermediate[tx + blockDim.x] = input[i + blockDim.x];
    }

    // Reduction tree
    for(uint32_t stride = 1; stride <= blockDim.x; stride *=2) {
        __syncthreads();
        // Reduce control divergence
        uint32_t index = (tx + 1) * 2 * stride - 1;
        if(index < SECTION_SIZE) {
            intermediate[index] += intermediate[index - stride];
        }
    }

    // Reverse tree
    for(uint32_t stride = SECTION_SIZE / 4; stride > 0; stride /= 2) {
        __syncthreads();
        uint32_t index = (tx + 1) * 2 * stride - 1;
        if(index + stride < SECTION_SIZE) {
            intermediate[index + stride] += intermediate[index];
        }
    }
    __syncthreads();
    if(i < N) {
        output[i] = intermediate[tx];
    }
    if(i < blockDim.x < N) {
        output[i + blockDim.x] = intermediate[tx + blockDim.x];
    }

}


void scan_v1_invok(float *input, float *output, uint32_t N) {
    assert(SECTION_SIZE <= 2048);
    assert(SECTION_SIZE == N);
    dim3 blockDim(N, 1, 1);
    dim3 gridDim(1, 1, 1);

    scan_v1<<<gridDim, blockDim>>>(input, output, N);
}