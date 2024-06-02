#include <iostream>
#include <cstdlib>
#include <chrono>

#include "convolution.hh"

int main(int argc, char **argv){

    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " (v0|v1|v2)" << std::endl;
        return 1;
    }

    std::string mode = argv[1];

    uint32_t width, height; // The width and height of the input 2D array.
    uint32_t r; // The radius of the convolution filter.
    

    FILE *fi;
    fi = fopen("data/input.dat", "rb");
    fread(&width, 1, sizeof(int), fi);
    fread(&height, 1, sizeof(int), fi);
    fread(&r, 1, sizeof(int), fi);

    std::cout << "width: " << width << ", height: " << height << ", r: " << r << std::endl;


    float *in = (float *)malloc(width * height * sizeof(float));
    float *out = (float *)malloc(width * height * sizeof(float));
    float *filter = (float *)malloc((2 * r + 1) * (2 * r + 1) * sizeof(float));

    fread(in, 1, width * height * sizeof(float), fi);
    fread(filter, 1, (2 * r + 1) * (2 * r + 1) * sizeof(float), fi);
    fclose(fi);

    if(mode == "v0"){
        std::cout << "The mode is v0" << std::endl;

        float *d_in, *d_out, *d_filter; 
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));
        cudaMalloc(&d_filter, (2 * r + 1) * (2 * r + 1) * sizeof(float));

        cudaMemcpy(d_in, in, width * height * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, filter, (2 * r + 1) * (2 * r + 1) * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 gridDim(ceil(float(width) / BLOCK_WIDTH), ceil(float(height) / BLOCK_WIDTH));

        convolution_v0<<<gridDim, blockDim>>>(d_in, d_filter, d_out, r, width, height);
        cudaDeviceSynchronize();
        cudaMemcpy(out, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        fi = fopen("data/ref.dat", "wb");
        fwrite(out, 1, width * height * sizeof(float), fi);
        fclose(fi);

        free(in); free(out); free(filter);
        cudaFree(d_in); cudaFree(d_out); cudaFree(d_filter);

    }
    else if(mode == "v1"){
        std::cout << "The mode is v1" << std::endl;

        
        float *d_in, *d_out;
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));

        cudaMemcpy(d_in, in, width * height * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH);
        dim3 gridDim(ceil(float(width) / BLOCK_WIDTH), ceil(float(height) / BLOCK_WIDTH));

        convolution_v1<<<gridDim, blockDim>>>(d_in, d_out, width, height);
        cudaDeviceSynchronize();
        cudaMemcpy(out, d_out, width * height * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        fi = fopen("data/convolution_v1.dat", "wb");
        fwrite(out, 1, width * height * sizeof(float), fi);
        fclose(fi);

        free(in); free(out); free(filter);
        cudaFree(d_in); cudaFree(d_out);
    }
    else if(mode == "v2"){
        std::cout << "The mode is v2" << std::endl;
    }
    else{
        std::cerr << "Invalid mode" << std::endl;
        return 1;
    }

    std::cout << "Done" << std::endl;
    return 0;
}