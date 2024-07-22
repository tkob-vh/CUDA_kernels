/*
 * convolution_v2.cu
 * A convolution kernel using cuda.
 * The convolution filter is a square matrix.
*/


#include "convolution.hh"
/*
 * The convolution kernel using constant memory and tiling.
 * Each CUDA thread laods a tile of the input 2D array into shared memory. During the convolution, some threads remain idle.
 * The input tile size is the same as the block size.
*/

__global__ void convolution_v2(float *N, float *P, int width, int height){
    int Col = blockIdx.x * OUT_TILE_WIDTH + threadIdx.x - FILTER_RADIUS; // The column index of the input 2D array.
    int Row = blockIdx.y * OUT_TILE_WIDTH + threadIdx.y - FILTER_RADIUS; // The row index of the input 2D array.

    // The shared memory for the input tile.
    __shared__ float N_s[IN_TILE_WIDTH][IN_TILE_WIDTH];
    if(Col >= 0 && Col < width && Row >= 0 && Row < height){
        N_s[threadIdx.y][threadIdx.x] = N[Row * width + Col];
    }else{
        N_s[threadIdx.y][threadIdx.x] = 0.0f;
    }
    __syncthreads();

    int tileCol = threadIdx.x - FILTER_RADIUS;
    int tileRow = threadIdx.y - FILTER_RADIUS;

    if(Col >= 0 && Col < width && Row >= 0 && Row < height){
        if(tileCol >= 0 && tileCol < OUT_TILE_WIDTH && tileRow >= 0 && tileRow < OUT_TILE_WIDTH){
            float Pvalue = 0.0f;
            for(int fRow = 0; fRow < 2 * FILTER_RADIUS + 1; fRow++){
                for(int fCol = 0; fCol < 2 * FILTER_RADIUS + 1; fCol++){
                    Pvalue += N_s[tileRow + fRow][tileCol + fCol] * F[fRow][fCol];
                }
            }
            P[Row * width + Col] = Pvalue;
        }
   }
}