/*
 * stencil_v3.cu
 * A 3D stencil kernel with shared memory and thread coarsening and register tiling.
 * The grid is 3D, while the block is 2D.
 * Improve the tile size and shared memory usage.
 */

#include "stencil.hh"



__global__ void stencil_v3(const float *in, float *out, int nx, int ny, int nz){
    int tx = threadIdx.x; int ty = threadIdx.y;

    int iStart = blockIdx.z * OUT_TILE_WIDTH2;
    int j = blockIdx.y * OUT_TILE_WIDTH2 + ty - 1;
    int k = blockIdx.x * OUT_TILE_WIDTH2 + tx - 1;

    float inPrev;
    __shared__ float inCurr_s[IN_TILE_WIDTH2][IN_TILE_WIDTH2];
    float inCurr;
    float inNext;

    //Initialize the shared memory(The first 2 tiles)
    if(iStart >= 1 && iStart < nz + 1 && j >= 0 && j < ny && k >= 0 && k < nx){
        // If the first tile is not out of bound.
        inPrev = in[(iStart - 1) * nx * ny + j * nx + k];
    }
    if(iStart >= 0 && iStart < nz && j >= 0 && j < ny && k >= 0 && k < nx){ 
        // If the second tile is not out of bound.
        inCurr = in[iStart * nx * ny + j * nx + k];
        inCurr_s[ty][tx] = inCurr;
    }
    
    for(int p = iStart; p < iStart + OUT_TILE_WIDTH2; p++){
        if(p >= -1 && p < nz - 1 && j >= 0 && j < ny && k >= 0 && k < nx){  
            // The third tile
            inNext = in[(p + 1) * nx * ny + j * nx + k];
        }
        __syncthreads();

        if(p >= 1 && p < nz - 1
            && j >= 1 && j < ny - 1
            && k >= 1 && k < nx - 1){ 
            // Skip the boundry.
            if(tx >= 1 && tx < IN_TILE_WIDTH2 - 1
                && ty >= 1 && ty < IN_TILE_WIDTH2 - 1){
                // Only compute the output tile region
                out[p * nx * ny + j * nx * k] = c0 * inCurr_s[ty][tx]
                                              + c1 * inCurr_s[ty][tx - 1]
                                              + c2 * inCurr_s[ty][tx + 1]
                                              + c3 * inCurr_s[ty - 1][tx]
                                              + c4 * inCurr_s[ty + 1][tx]
                                              + c5 * inPrev
                                              + c6 * inNext;
            }
        }

        __syncthreads();
        //Update the input tiles in the shared memory.
        inPrev = inCurr;
        inCurr = inNext;
        inCurr_s[ty][tx] = inNext;
    }
    


}

void stencil_v3_invok(uint32_t nx, uint32_t ny, uint32_t nz,
                        float *in, float *out) {

    dim3 blockDim(IN_TILE_WIDTH2, IN_TILE_WIDTH2, 1);
    dim3 gridDim(ceil((float) nx / IN_TILE_WIDTH2),
                ceil((float) ny / IN_TILE_WIDTH2),
                ceil((float) nz / OUT_TILE_WIDTH2));

    stencil_v3<<<blockDim, gridDim>>>(in, out, nx, ny, nz);
    cudaDeviceSynchronize();
    }