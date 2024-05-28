/*
 * gemmv0.cu
 * The code is the implementation of the matrix multiplication using CUDA.
 * The Matrix M, N, and P are Square matrices of size Width x Width.
 * The size of the shared memory is TILE_WIDTH x TILE_WIDTH.
 * The Block size should be equal to TILE_WIDTH x TILE_WIDTH.
*/


#include<cuda_runtime.h>
#include<gemm.hh>

__global__ void matrixMul0(float *M, float *N, float *P, int Width){

    // Use shared memory to store the sub-matrices of M and N, which improves compute density and reduces the number of global memory accesses.
    __shared__ float Mds[TILE_WIDTH][TILE_WIDTH];
    __shared__ float Nds[TILE_WIDTH][TILE_WIDTH];

    int bx = blockIdx.x; int by = blockIdx.y;
    int tx = threadIdx.x; int ty = threadIdx.y;

    // The Row and Column of the current thread which is responsible for computing its Pvalue[Col][Row]
    int Row = by * TILE_WIDTH + ty;
    int Col = bx * TILE_WIDTH + tx;

    float Pvalue = 0; // Store the element of the resulting matrix
    for(int phase = 0; phase < ceil(Width/(float)TILE_WIDTH); phase++){
        // Boundary check
        if(Row < Width && phase * TILE_WIDTH + tx < Width)
            Mds[ty][tx] = M[Row * Width + phase * TILE_WIDTH + tx];
        else
            Mds[ty][tx] = 0.0f;

        if(Col < Width && phase * TILE_WIDTH + ty < Width)
            Nds[ty][tx] = N[(phase * TILE_WIDTH + ty) * Width + Col];
        else
            Nds[ty][tx] = 0.0f;
        __syncthreads();

        for(int k = 0; k < TILE_WIDTH; k++){
            Pvalue += Mds[ty][k] * Nds[k][tx];
        }
        __syncthreads();

    }

    if(Row < Width && Col < Width)
        P[Row * Width + Col] = Pvalue;

}