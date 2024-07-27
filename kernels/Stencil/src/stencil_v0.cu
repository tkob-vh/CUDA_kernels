/*
 * stencil_v0.cu
 * A 3D stencil kernel.
 */

#include "stencil.hh"


__global__ void stencil_v0(const float *in, float *out, int nx, int ny, int nz){
    int gx = blockIdx.x * blockDim.x + threadIdx.x;
    int gy = blockIdx.y * blockDim.y + threadIdx.y;
    int gz = blockIdx.z * blockDim.z + threadIdx.z;

    if(gx > 0 && gx < nx-1 && gy > 0 && gy < ny-1 && gz > 0 && gz < nz-1){
        out[gx + gy*nx + gz*nx*ny] =    c0 * in[gx + gy*nx + gz*nx*ny]
                                    +   c1 * in[(gx-1) + gy*nx + gz*nx*ny]
                                    +   c2 * in[(gx+1) + gy*nx + gz*nx*ny]
                                    +   c3 * in[gx + (gy-1)*nx + gz*nx*ny]
                                    +   c4 * in[gx + (gy+1)*nx + gz*nx*ny]
                                    +   c5 * in[gx + gy*nx + (gz-1)*nx*ny]
                                    +   c6 * in[gx + gy*nx + (gz+1)*nx*ny];
    }
}

void stencil_v0_invok(uint32_t nx, uint32_t ny, uint32_t nz,
                        float *in, float *out) {

        dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
        dim3 gridDim(ceil((float)nx / blockDim.x),
                    ceil((float)ny / blockDim.y),
                    ceil((float)nz / blockDim.z));

        stencil_v0<<<blockDim, gridDim>>>(in, out, nx, ny, nz);

        cudaDeviceSynchronize();
}