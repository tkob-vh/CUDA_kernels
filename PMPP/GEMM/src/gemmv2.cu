/*
 * gemmv2.cu
 * The Matrix M and N don't have to be square matrices.
 * The size of the shared memory is not determined until runtime.
*/

#include <cuda_runtime.h>


__global__ void matrixMul2(float *M, float *N, float *P, int r, int s, int t, unsigned int Mds_sz, unsigned int Nds_sz){
    
    extern __shared__ float Mds_Nds[];

    float *Mds = Mds_Nds;
    float *Nds = Mds_Nds + Mds_sz;

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * blockDim.y + ty;
    int Col = bx * blockDim.x + tx;

    float Pvalue = 0;
    for(int phase = 0; phase < ceil(s/(float)Nds_sz) && phase < ceil(s/(float)Mds_sz); phase ++){
        if(Row < r && (phase * Nds_sz + tx) < s)
            Mds[ty * Mds_sz + tx] = M[Row * s + phase * Nds_sz + tx];
        else
            Mds[ty * Mds_sz + tx] = 0.0f;
        
        if(Col < t && (phase * Mds_sz + ty) < s)
            Nds[ty * Mds_sz + tx] = N[(phase * Mds_sz + ty) * t + Col];
        else
            Nds[ty * Mds_sz + tx] = 0.0f;

        __syncthreads();

        for(int k = 0; k < Mds_sz; k++){
            Pvalue += Mds[ty * Mds_sz + k] * Nds[k * Mds_sz + tx];
        }
        __syncthreads();
    }

    if(Row < r && Col < t)
        P[Row * t + Col] = Pvalue;
}