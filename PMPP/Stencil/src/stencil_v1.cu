/*
 * stencil_v1.cu
 * A 3D stencil kernel with shared memory.
 */

#include "stencil.hh"

// The IN_TILE_WIDTH should be the same as the block width.
#define IN_TILE_WIDTH 8
#define OUT_TILE_WIDTH 6

__global__ void stencil_v1(const float *in, float *out, int nx, int ny, int nz){
    int tx = threadIdx.x; int ty = threadIdx.y; int tz = threadIdx.z;

    int gx = blockIdx.x * OUT_TILE_WIDTH + tx - 1;
    int gy = blockIdx.y * OUT_TILE_WIDTH + ty - 1;
    int gz = blockIdx.z * OUT_TILE_WIDTH + tz - 1;


    __shared__ float in_s[IN_TILE_WIDTH][IN_TILE_WIDTH][IN_TILE_WIDTH];

    //Load the input tile to the shared memory.
    if(gx >= 0 && gx < nx && gy >= 0 && gy < ny && gz >= 0 && gz < nz){
        in_s[tz][ty][tx] = in[gz * nx * ny + gy * nx + nx];
    }
    __syncthreads();

    if(gx > 0 && gx < nx - 1 && gy > 0 && gy < ny - 1 && gz > 0 && gz < nz - 1){    //Only the non-edge elements need to be computed.
        if(tx > 0 && tx < IN_TILE_WIDTH - 1 && ty > 0 && ty < IN_TILE_WIDTH - 1 && tz > 0 && tz < IN_TILE_WIDTH - 1){   // Only some of threads(The OUT_TILE_WIDTH region) in a block need to be computed.
            out[gz * nx * ny + gy * nx + gx] = c0 * in_s[tz][ty][tx]
                                             + c1 * in_s[tz][ty][tx - 1]
                                             + c2 * in_s[tz][ty][tx + 1]
                                             + c3 * in_s[tz][ty - 1][tx]
                                             + c4 * in_s[tz][ty + 1][tx]
                                             + c5 * in_s[tz - 1][ty][tx]
                                             + c6 * in_s[tz + 1][ty][tx];
        }
    }
}
