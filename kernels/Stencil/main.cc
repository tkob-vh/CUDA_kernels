#include <iostream>
#include <string>
#include <cuda_runtime.h>
#include <filesystem>
#include <fstream>
#include <chrono>
#include "stencil.hh"

int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Usage:" << argv[0] << " (v0|v1|v2)" << std::endl;
        return 1;
    }

    std::string version = argv[1];
    uint32_t nx, ny, nz;

    FILE *fi;
    std::filesystem::path input_path = "data/input.dat";
    std::ifstream input_file(input_path, std::ios::binary);
    input_file.read(reinterpret_cast<char *>(&nx), sizeof(nx));
    input_file.read(reinterpret_cast<char *>(&ny), sizeof(ny));
    input_file.read(reinterpret_cast<char *>(&nz), sizeof(nz));

    std::cout << "nx: " << nx << ", ny: " << ny << ", nz: " << nz << std::endl;

    float *in = new float[nx * ny * nz];
    float *out = new float[nx * ny * nz];

    input_file.read(reinterpret_cast<char *>(in), nx * ny * nz * sizeof(float));

    if(version == "v0"){
        std::cout << "The version is v0" << std::endl;

        float *d_in, *d_out;

        auto t1 = std::chrono::steady_clock::now();

        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float),
                    cudaMemcpyHostToDevice);

        stencil_v0_invok(nx, ny, nz, d_in, d_out);

        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/stencil_v0.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        nx * ny * nz * sizeof(float));


        cudaFree(d_in); cudaFree(d_out);
    }
    else if(version == "v1"){
        std::cout << "The version is v1" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        float *d_in, *d_out;
        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float),
                    cudaMemcpyHostToDevice);

        stencil_v1_invok(nx, ny, nz, d_in, d_out);

        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;
        
        std::filesystem::path output_path = "data/stencil_v1.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        nx * ny * nz * sizeof(float));       

        cudaFree(d_in); cudaFree(d_out);
    }
    else if(version == "v2"){
        std::cout << "The version is v2" << std::endl;

        auto t1 = std::chrono::steady_clock::now();

        float *d_in, *d_out;
        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float),
                    cudaMemcpyHostToDevice);

        stencil_v2_invok(nx, ny, nz, d_in, d_out);
        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                        <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/stencil_v2.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        nx * ny * nz * sizeof(float));

        cudaFree(d_in); cudaFree(d_out);

    }
    else if(version == "v3"){
        std::cout << "The version is v3" << std::endl;

        auto t1 = std::chrono::steady_clock::now();

        float *d_in, *d_out;
        cudaMalloc(&d_in, nx * ny * nz * sizeof(float));
        cudaMalloc(&d_out, nx * ny * nz * sizeof(float));

        cudaMemcpy(d_in, in, nx * ny * nz * sizeof(float),
                    cudaMemcpyHostToDevice);

        stencil_v3_invok(nx, ny, nz, d_in, d_out);
        
        cudaMemcpy(out, d_out, nx * ny * nz * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/stencil_v3.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        nx * ny * nz * sizeof(float));

        cudaFree(d_in); cudaFree(d_out);

    }
    else{
        std::cerr << "Invalid version" << std::endl;
        return 1;
    }

    free(in); free(out);
    std::cout << "Done" << std::endl;
    return 0;

}