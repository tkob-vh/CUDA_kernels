#include <iostream>
#include <cstdlib>
#include <chrono>
#include <filesystem>
#include <fstream>

#include "convolution.hh"

int main(int argc, char **argv){

    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " (v0|v1|v2)" << std::endl;
        return 1;
    }

    std::string version = argv[1];

    uint32_t width, height; // The width and height of the input 2D array.
    uint32_t r; // The radius of the convolution filter.
    
    std::filesystem::path input_path = "data/input.dat";
    std::ifstream input_file(input_path, std::ios::binary);
    input_file.read(reinterpret_cast<char *>(&width), sizeof(width));
    input_file.read(reinterpret_cast<char *>(&height), sizeof(height));
    input_file.read(reinterpret_cast<char *>(&r), sizeof(r));

    std::cout << "width: " << width << ", height: " << height 
            << ", r: " << r << std::endl;

    float *in = new float[width * height]{};
    float *out = new float[width * height]{};
    float *filter = new float[(2 * r + 1) * (2 * r + 1)]{};

    input_file.read(reinterpret_cast<char *>(in),
                    width * height * sizeof(float));
    input_file.read(reinterpret_cast<char *>(filter), 
                    (2 * r + 1) * (2 * r + 1) * sizeof(float));

    if(version == "v0"){
        std::cout << "The version is v0" << std::endl;

        float *d_in, *d_out, *d_filter; 
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));
        cudaMalloc(&d_filter, (2 * r + 1) * (2 * r + 1) * sizeof(float));

        cudaMemcpy(d_in, in, width * height * sizeof(float),
                    cudaMemcpyHostToDevice);
        cudaMemcpy(d_filter, filter, (2 * r + 1) * (2 * r + 1) * sizeof(float),
                    cudaMemcpyHostToDevice);
        
        convolution_v0_invok(width, height, r, d_in, d_out, d_filter);

        cudaMemcpy(out, d_out, width * height * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/convolution_v0.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        width * height * sizeof(float));

        cudaFree(d_in); cudaFree(d_out); cudaFree(d_filter);

    }
    else if(version == "v1"){
        std::cout << "The version is v1" << std::endl;

        
        float *d_in, *d_out;
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));

        cudaMemcpy(d_in, in, width * height * sizeof(float),
                    cudaMemcpyHostToDevice);

        convolution_v1_invok(width, height, d_in, d_out);

        cudaMemcpy(out, d_out, width * height * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/convolution_v1.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        width * height * sizeof(float));

        cudaFree(d_in); cudaFree(d_out);
    }
    else if(version == "v2"){
        std::cout << "The version is v2" << std::endl;
             
        float *d_in, *d_out;
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_in, width * height * sizeof(float));
        cudaMalloc(&d_out, width * height * sizeof(float));

        cudaMemcpy(d_in, in, width * height * sizeof(float),
                    cudaMemcpyHostToDevice);

        convolution_v2_invok(width, height, d_in, d_out);

        cudaMemcpy(out, d_out, width * height * sizeof(float),
                    cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/convolution_v2.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(out),
                        width * height * sizeof(float));

        cudaFree(d_in); cudaFree(d_out);
    }
    else{
        std::cerr << "Invalid version" << std::endl;
        return 1;
    }

    delete [] in;
    delete [] out;
    delete [] filter;
    return 0;
}