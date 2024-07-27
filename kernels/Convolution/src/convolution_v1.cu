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
__global__ void convolution_v1(float *N, float *P, int width, int height){
    int Col = blockIdx.x * blockDim.x + threadIdx.x;
    int Row = blockIdx.y * blockDim.y + threadIdx.y;

    float Pvalue = 0.0f;
    for(int fRow = -FILTER_RADIUS; fRow <= FILTER_RADIUS; fRow++){
        for(int fCol = -FILTER_RADIUS; fCol <= FILTER_RADIUS; fCol++){
            if(Col + fCol >= 0 && Col + fCol < width && Row + fRow >= 0 && Row + fRow < height){
                Pvalue += N[(Row + fRow) * width + Col + fCol] * F[fRow + FILTER_RADIUS][fCol + FILTER_RADIUS];
            }
        }
    }
    if(Col < width && Row < height)
        P[Row * width + Col] = Pvalue;

}


void convolution_v1_invok(uint32_t width, uint32_t height,
                            float *in, float *out) {
    dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
    dim3 gridDim(ceil(float(width) / BLOCK_WIDTH),
                ceil(float(height) / BLOCK_WIDTH));

    convolution_v1<<<gridDim, blockDim>>>(in, out, width, height);
    cudaDeviceSynchronize();
}