/* 
 * histogram_v0.cu
 * A navie parallel histogram kernel.
 */
#include "histogram.hh"

__global__ void histogram_v0(char *data, unsigned int length, unsigned int *histo){
    unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
    if(i < length){
        int alphabet_position = data[i] - 'a';
        if(alphabet_position >= 0 && alphabet_position < 26){
            atomicAdd(&histo[alphabet_position/4], 1);
        }
    }
}