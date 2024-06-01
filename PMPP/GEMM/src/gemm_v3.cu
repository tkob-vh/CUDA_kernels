/*
 * gemmv3.cu
 * The Matrix M and N don't have to be square matrices.
 * The size of the shared memory is not determined until runtime.
 * Use thread coarsening to improve the performance.
*/

#include "gemm.hh"

#define COARSE_FACTOR 4


__global__ void matrixMul3(float *M, float *N, float *P, int r, int s, int t, unsigned int Mds_sz, unsigned int Nds_sz){
    int tile_width = blockDim.x; 
    extern __shared__ char Mds_Nds[];

    float *Mds = (float *) Mds_Nds;
    float *Nds = (float *) (Mds_Nds + Mds_sz);

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * tile_width + ty;
    int colStart = bx * tile_width * COARSE_FACTOR + tx;

    float Pvalue[COARSE_FACTOR];
    for(int i = 0; i < COARSE_FACTOR; i++)
        Pvalue[i] = 0.0f;

    for(int phase = 0; phase < ceil(s/(float)tile_width); phase ++){
        if(Row < r && (phase * tile_width + tx) < s)
            Mds[ty * tile_width + tx] = M[Row * s + phase * tile_width + tx];
        else
            Mds[ty * tile_width + tx] = 0.0f;
        
        for(int i = 0; i < COARSE_FACTOR; i++){
            int Col = colStart + i * tile_width;
            if(Col < t && (phase * tile_width + ty) < s)
                Nds[ty * tile_width + tx] = N[(phase * tile_width + ty) * t + Col];
            else
                Nds[ty * tile_width + tx] = 0.0f;

            __syncthreads();


            for(int k = 0; k < tile_width; k++){
                Pvalue[i] += Mds[ty * tile_width + k] * Nds[k * tile_width + tx];
            }
            __syncthreads();
        }
    }

    for(int i = 0; i < COARSE_FACTOR; i++){
        int Col = colStart + i * tile_width;
        if(Row < r && Col < t)
            P[Row * t + Col] = Pvalue[i];
    }
}