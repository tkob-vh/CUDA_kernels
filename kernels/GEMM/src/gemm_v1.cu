/*
 * gemmv1.cu
 * The Matrix M and N don't have to be square matrices. 
 * The size of the shared memory is TILE_WIDTH x TILE_WIDTH.
 * The Block size should be equal to TILE_WIDTH x TILE_WIDTH.
*/

#include<gemm.hh>


__global__ void matrixMul1(float *M, float *N, float *P, int r, int s, int t){
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0;

    for(int phase = 0; phase < ceil(s/(float)TILE_WIDTH); phase++){
        if(Row < r && (phase * TILE_WIDTH + tx) < s)
            Mds[ty][tx] = M[Row * s + phase * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if(Col < t && (phase * TILE_WIDTH + ty) < s)
            Nds[ty][tx] = N[(phase * TILE_WIDTH + ty) * t + Col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }

    if(Row < r && Col < t)
        P[Row * t + Col] = Pvalue;
}