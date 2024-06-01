/*
 * convolution_v1.cu
 * A convolution kernel using cuda.
 * The convolution filter is a square matrix.
*/
#include "convolution.hh"
/*
 * The convolution kernel using constant memory.
 * Each CUDA thread computes the convolution of a single pixel.
 * N: The input 2D array.
 * F: The convolution filter(convolution kernel).
 * P: The output 2D array.
 * r: The radius of the convolution filter.
 * width: The width of the input(output) 2D array.
 * height: The height of the input(output) 2D array.
*/
__global__ void convolution_v1(float *N, float *P, int r, int width, int height){
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    for(int fRow = -r; fRow <= r; fRow++){
        for(int fCol = -r; fCol <= r; fCol++){
            if(Col + fCol >= 0 && Col + fCol < width && Row + fRow >= 0 && Row + fRow < height){
                Pvalue += N[(Row + fRow) * width + Col + fCol] * F[fRow + r][fCol + r];
            }
        }
    }
    if(Col < width && Row < height)
        P[Row * width + Col] = Pvalue;

}