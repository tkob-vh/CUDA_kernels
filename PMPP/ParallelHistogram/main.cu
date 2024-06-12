#include <iostream>
#include <chrono>
#include <string>
#include "histogram.hh"

int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << "(v0|v1|v2)" << std::endl;
        return -1;
    }
    std::string mode = argv[1];

    int length;
    unsigned int histo[BUCKET];

    FILE *fi;

    fi = fopen("data/input.dat", "rb");
    fread(&length, 1, sizeof(int), fi);
    std::cout << "The length of the random letters: " << length << std::endl;

    char *in = (char *)malloc(length * sizeof(char));

    fread(in, sizeof(char), length, fi);
    fclose(fi);

    if(mode == "v0"){
        std::cout << "The mode is v0" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, BUCKET * sizeof(unsigned int));

        cudaMemcpy(in_d, in, length * sizeof(char), cudaMemcpyHostToDevice);

        dim3 blockDim(1024, 1, 1);
        dim3 gridDim(ceil(float(length) / blockDim.x), 1, 1);

        histogram_v0<<<blockDim, gridDim>>>(in_d, length, histo_d);
        cudaDeviceSynchronize();

        cudaMemcpy(histo, histo_d, BUCKET * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        for(int i = 0; i < BUCKET; i++){
            std::cout << histo[i] << std::endl;
        } 

        free(in);
        cudaFree(in_d); cudaFree(histo_d);
    }
    else if(mode == "v1") {
        std::cout << "The mode is v1" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, BUCKET * sizeof(unsigned int));

        cudaMemcpy(in_d, in, length * sizeof(char), cudaMemcpyHostToDevice);

        dim3 blockDim(1024, 1, 1);
        dim3 gridDim(ceil(float(length) / blockDim.x), 1, 1);

        histogram_v1<<<blockDim, gridDim>>>(in_d, length, histo_d);
        cudaMemcpy(histo, histo_d, BUCKET * sizeof(unsigned int), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        std::cout << d1 << std::endl;

        for(int i = 0; i < BUCKET; i++){
            std::cout << histo[i] << std::endl;
        }
        free(in);
        cudaFree(in_d); cudaFree(histo_d);
    }
    else{
        std::cout << "The mode is illegal" << std::endl;
        free(in);
        return -1;
    }

    return 0;
}