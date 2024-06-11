#include <cuda_runtime.h>
#define BUCKET 7

__global__ void histogram_v0(char *data, unsigned int length, unsigned int *histo);