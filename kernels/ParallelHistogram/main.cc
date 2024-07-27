#include <iostream>
#include <chrono>
#include <string>
#include <filesystem>
#include <fstream>

#include "histogram.hh"

int main(int argc, char **argv){
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << "(v0|v1|v2)" << std::endl;
        return -1;
    }
    std::string version = argv[1];

    int length;
    unsigned int histo[NUM_BINS];


    std::filesystem::path input_path = "data/input.dat";
    std::ifstream input_file(input_path, std::ios::binary);
    input_file.read(reinterpret_cast<char *>(&length), sizeof(int));

    std::cout << "The length of the random letters: " << length << std::endl;

    char *in = new char[length];

    input_file.read(reinterpret_cast<char *>(in), length *sizeof(char));

    if(version == "v0"){
        std::cout << "The version is v0" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));

        cudaMemcpy(in_d, in, length * sizeof(char), cudaMemcpyHostToDevice);

        histogram_v0_invok(length, in_d, histo_d);

        cudaMemcpy(histo, histo_d, NUM_BINS * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        for(int i = 0; i < NUM_BINS; i++){
            std::cout << histo[i] << std::endl;
        } 

        cudaFree(in_d); cudaFree(histo_d);
    }
    else if(version == "v1") {
        std::cout << "The version is v1" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));

        cudaMemcpy(in_d, in, length * sizeof(char), cudaMemcpyHostToDevice);

        histogram_v1_invok(length, in_d, histo_d);
        cudaMemcpy(histo, histo_d, NUM_BINS * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        for(int i = 0; i < NUM_BINS; i++){
            std::cout << histo[i] << std::endl;
        }
        cudaFree(in_d); cudaFree(histo_d);
    }
    else if(version == "v2") {
        std::cout << "The version is v2" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int  *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));

        histogram_v2_invok(length, in_d, histo_d);
        cudaMemcpy(histo, histo_d, NUM_BINS * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;
        for(int i = 0; i < NUM_BINS; i++) {
            std::cout << histo[i] << std::endl;
        }

        cudaFree(in_d); cudaFree(histo_d);
    }
    else if(version == "v3") {
        std::cout << "The version is v3" << std::endl;

        auto t1 = std::chrono::steady_clock::now();
        char *in_d;
        unsigned int  *histo_d;

        cudaMalloc(&in_d, length * sizeof(char));
        cudaMalloc(&histo_d, NUM_BINS * sizeof(unsigned int));

        histogram_v3_invok(length, in_d, histo_d);
        cudaMemcpy(histo, histo_d, NUM_BINS * sizeof(unsigned int),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();

        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;
        for(int i = 0; i < NUM_BINS; i++) {
            std::cout << histo[i] << std::endl;
        }

        cudaFree(in_d); cudaFree(histo_d);
    }

    else{
        std::cout << "The version is illegal" << std::endl;
        return -1;
    }


    free(in);
    return 0;
}