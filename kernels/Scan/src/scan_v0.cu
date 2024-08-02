#include "scan.hh"

/*
 * \file scan_v0.cu
 * \Description: A naive parallel scan using Kogge-Stone algorithm.
 * Compute complexity: O(N * log2(N)).
*/


__global__ void scan_v0(float *input, float *output, uint32_t N) {
    __shared__ float intermediate[SECTION_SIZE];

    uint32_t tx = threadIdx.x;
    uint32_t i = blockDim.x * blockIdx.x + tx;
    if(i < N) {
        intermediate[tx] = input[i];
    }
    else {
        intermediate[tx] = 0.0f;
    }

    for(uint32_t stride = 1; stride < blockDim.x; stride *= 2) {
        __syncthreads();
        float tmp;

        // If stride > tx, it means the thread has accumulated all the required
        // input values for its position and the thread not longer needs to be 
        // active.
        if(tx >= stride) {
            tmp = intermediate[tx] + intermediate[tx - stride];
        }
        // A write-after-read dependency constraint.
        __syncthreads();
        if(tx >= stride) {
            intermediate[tx] = tmp;
        }
    }
    if(i < N) {
        output[i] = intermediate[tx];
    }
}


void scan_v0_invok(float *input, float *output, uint32_t N) {
    assert(SECTION_SIZE <= 1024);
    assert(SECTION_SIZE == N);
    dim3 blockDim(SECTION_SIZE, 1, 1);
    dim3 gridDim(1, 1, 1);   

    scan_v0<<<gridDim, blockDim>>>(input, output, N);
}