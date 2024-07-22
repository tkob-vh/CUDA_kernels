/* 
 * histogram_v2.cu
 * Tricks: Shared Memory, Privatization and Thread Coarsening.
 */
#include "histogram.hh"

__global__ void histogram_v2(char *data, unsigned int length,
                                unsigned int *histo){

    __shared__ unsigned int histo_s[NUM_BINS];
    // if(threadIdx.x < NUM_BINS)
    //     histo_s[threadIdx.x] = 0u;
    // Initialize the histo in shared memory(Robust version)
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    unsigned int tid = blockDim.x * blockIdx.x * CORASE_SIZE + threadIdx.x;
    for(unsigned int i = tid; i < min(length, tid + blockDim.x * 
        CORASE_SIZE); i += blockDim.x) {
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26) {
            atomicAdd(&histo_s[alphabet_position/4], 1);
        }
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x) {
        unsigned int binValue = histo_s[bin];
        if(binValue > 0) {
            atomicAdd(&histo[bin], binValue);
        }
    }
}