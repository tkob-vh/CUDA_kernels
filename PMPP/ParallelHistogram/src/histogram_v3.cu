/* 
 * histogram_v3.cu
 * Tricks: Shared Memory, Privatization, Thread Coarsening and Aggregation.
 */
#include "histogram.hh"

__global__ void histogram_v3(char *data, unsigned int length,
                                unsigned int *histo){

    __shared__ unsigned int histo_s[NUM_BINS];
    // if(threadIdx.x < NUM_BINS)
    //     histo_s[threadIdx.x] = 0u;
    // Initialize the histo in shared memory(Robust version)
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;

    unsigned int accumulator = 0;
    int prevBinIdx = -1;
    __syncthreads();
        

    unsigned int tid = blockDim.x * blockIdx.x * CORASE_SIZE + threadIdx.x;
    for(unsigned int i = tid; i < min(length, tid + blockDim.x * 
            CORASE_SIZE); i += blockDim.x) {
        int alphabet_position = data[i] - 'a';
        int bin = alphabet_position / 4;
        if(bin == prevBinIdx) {
            accumulator++;
        }
        else {
            if(accumulator > 0) {
                atomicAdd(&histo_s[prevBinIdx], accumulator);
            }
            accumulator = 1;
            prevBinIdx = bin;
        }
    }
    if(accumulator > 0) {
        atomicAdd(&histo_s[prevBinIdx], accumulator);
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if(binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}