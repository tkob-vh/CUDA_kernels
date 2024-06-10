#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <chrono>
#include "stencil.hh"

int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Usage:" << argv[0] << " (v0|v1|v2)" << std::endl;
        return 1;
    }

    std::string mode = argv[1];
    uint32_t nx, ny, nz;

    FILE *fi;
    fi = fopen("data/input.dat", "rb");
    fread(&nx, 1, sizeof(uint32_t), fi);
    fread(&ny, 1, sizeof(uint32_t), fi);
    fread(&nz, 1, sizeof(uint32_t), fi);
    std::cout << "nx: " << nx << ", ny: " << ny << ", nz: " << nz << std::endl;

    float *in = (float *)malloc(nx * ny * nz * sizeof(float));
    float *out = (float *)malloc(nx * ny * nz * sizeof(float));

    fread(in, 1, nx * ny * nz * sizeof(float), fi);
    fclose(fi);

    if(mode == "v0"){
        std::cout << "The mode is v0" << std::endl;

        float *d_in, *d_out;

        auto t1 = std::chrono::steady_clock::now();

        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);


        dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
        dim3 gridDim(ceil((float)nx / blockDim.x),ceil((float)ny / blockDim.y), ceil((float)nz / blockDim.z));

        stencil_v0<<<blockDim, gridDim>>>(d_in, d_out, nx, ny, nz);

        cudaDeviceSynchronize();

        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        fi = fopen("data/stencil_v0.dat", "wb");
        fwrite(out, 1, nx * ny * nz * sizeof(float), fi);
        fclose(fi);

        free(in); free(out);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if(mode == "v1"){
        std::cout << "The mode is v1" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        float *d_in, *d_out;
        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(IN_TILE_WIDTH, IN_TILE_WIDTH, IN_TILE_WIDTH);
        dim3 gridDim(ceil((float)nx / blockDim.x), ceil((float)ny / blockDim.y), ceil((float)nz / blockDim.z));

        stencil_v1<<<blockDim, gridDim>>>(d_in, d_out, nx, ny, nz);

        cudaDeviceSynchronize();

        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        fi = fopen("data/stencil_v1.dat", "wb");
        fwrite(out, 1, nx * ny * nz * sizeof(float), fi);
        fclose(fi);

        free(in); free(out);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if(mode == "v2"){
        std::cout << "The mode is v2" << std::endl;

        auto t1 = std::chrono::steady_clock::now();

        float *d_in, *d_out;
        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(IN_TILE_WIDTH2, IN_TILE_WIDTH2, 1);
        dim3 gridDim(ceil((float) nx / IN_TILE_WIDTH2), ceil((float) ny / IN_TILE_WIDTH2), ceil((float) nz / OUT_TILE_WIDTH2));

        stencil_v2<<<blockDim, gridDim>>>(d_in, d_out, nx, ny, nz);
        cudaDeviceSynchronize();
        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        fi = fopen("data/stencil_v2.dat", "wb");
        fwrite(out, 1, nx * ny * nz * sizeof(float), fi);
        fclose(fi);

        free(in); free(out);
        cudaFree(d_in); cudaFree(d_out);

    }
    else{
        std::cerr << "Invalid mode" << std::endl;
        return 1;
    }

    std::cout << "Done" << std::endl;
    return 0;

}