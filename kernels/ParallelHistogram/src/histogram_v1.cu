/* 
 * histogram_v1.cu
 * Tricks: Shared Memory, Privatization.
 */
#include "histogram.hh"

__global__ void histogram_v1(char *data, unsigned int length, unsigned int *histo){
    __shared__ unsigned int histo_s[NUM_BINS];
    // if(threadIdx.x < NUM_BINS)
    //     histo_s[threadIdx.x] = 0u;
    // Initialize the histo in shared memory(Robust version)
    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x)
        histo_s[bin] = 0u;
    __syncthreads();

    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&histo_s[alphabet_position/4], 1);
        }
    }
    __syncthreads();

    for(unsigned int bin = threadIdx.x; bin < NUM_BINS; bin += blockDim.x){
        unsigned int num = histo_s[bin];
        if(num > 0) atomicAdd(&histo[bin], num);
    }
}


void histogram_v1_invok(int length, char *in, unsigned int *histo) {
    dim3 blockDim(1024, 1, 1);
    dim3 gridDim(ceil(float(length) / blockDim.x), 1, 1);

    histogram_v1<<<gridDim, blockDim>>>(in, length, histo);
}