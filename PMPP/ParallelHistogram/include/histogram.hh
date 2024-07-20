#include <cuda_runtime.h>
#define BUCKET 7

__global__ void histogram_v0(char *data, unsigned int length, unsigned int *histo);
__global__ void histogram_v1(char *data, unsigned int length, unsigned int *histo);

#define CORASE_SIZE 16
__global__ void histogram_v2(char *data, unsigned int length, unsigned int *histo);