#include <cuda_runtime.h>
#define NUM_BINS 7

__global__ void histogram_v0(char *data, unsigned int length,
                            unsigned int *histo);
void histogram_v0_invok(int length, char *in, unsigned int *histo);

__global__ void histogram_v1(char *data, unsigned int length,
                            unsigned int *histo);
void histogram_v1_invok(int length, char *in, unsigned int *histo);

#define CORASE_SIZE 16
__global__ void histogram_v2(char *data, unsigned int length,
                            unsigned int *histo);
void histogram_v2_invok(int length, char *in, unsigned int *histo);


__global__ void histogram_v3(char *data, unsigned int length,
                            unsigned int *histo);
void histogram_v3_invok(int length, char *in, unsigned int *histo);
