#include <iostream>
#include <cstdlib>
#include <chrono>
#include <string>
#include <mkl.h>
#include <filesystem>
#include <fstream>

#include "gemm.hh"

int main(int argc, char **argv){
    // Check the number of arguments
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " (cpu|v0|v1|v2)" 
                << std::endl;
        return -1;
    }

    std::string version = argv[1]; 

    uint32_t n1, n2, n3;

    // Read the input data

    std::filesystem::path input_path = "data/input.dat";
    std::ifstream input_file(input_path, std::ios::binary);
    input_file.read(reinterpret_cast<char *>(&n1), sizeof(n1));
    input_file.read(reinterpret_cast<char *>(&n2), sizeof(n2));
    input_file.read(reinterpret_cast<char *>(&n3), sizeof(n3));

    printf("n1: %u, n2: %u, n3: %u\n", n1, n2, n3);
    std::cout << "n1: " << n1 << ", n2: " << n2 << ", n3: " << n3 << std::endl;

    float *a = new float[n1 * n2];
    float *b = new float[n2 * n3];
    float *c = new float[n1 * n3];


    input_file.read(reinterpret_cast<char *>(a), n1 * n2 *sizeof(float));
    input_file.read(reinterpret_cast<char *>(b), n2 * n3 *sizeof(float));

    if(version == "cpu"){
        std::cout << "The version is cpu" << std::endl;
        // Initialize c

        // Measure the time
        auto t1 = std::chrono::steady_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n3, n2, 
                    1.0, a, n2, b, n3, 0.0, c, n3);
        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                                <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/ref.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(float));

    }
    else if(version == "v0"){
        std::cout << "The version is v0" << std::endl;

        float *d_a, *d_b, *d_c;
        auto t1 = std::chrono::steady_clock::now();

        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);
        
        gemm_v0_invok(n1, n2, n3, d_a, d_b, d_c);

        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);


        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                                <std::chrono::milliseconds>(t2 - t1).count();
        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/gemmv0.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(float));

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    else if(version == "v1"){
        std::cout << "The version is v1." << std::endl;
        float *d_a, *d_b, *d_c;

        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);

        gemm_v1_invok(n1, n2, n3, d_a, d_b, d_c);
        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                                <std::chrono::milliseconds>(t2 - t1).count();


        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/gemmv1.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(float));        

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    else if(version == "v2"){
       
        std::cout << "The version is gemmv2" << std::endl;
        float *d_a, *d_b, *d_c;
        
        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);
        
        gemm_v2_invok(n1, n2, n3, d_a, d_b, d_c);

        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);

        

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();


        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/gemmv2.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(float));        

        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    else if(version == "v3"){
        std::cout << "The version is v3" << std::endl;
        float *d_a, *d_b, *d_c;
        

        auto t1 = std::chrono::steady_clock::now();
        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);

        gemm_v3_invok(n1, n2, n3, d_a, d_b, d_c); 
        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);

        

        auto t2 = std::chrono::steady_clock::now();
        int duration = std::chrono::duration_cast
                            <std::chrono::milliseconds>(t2 - t1).count();

        std::cout << "Duration: " << duration << " ms" << std::endl;

        std::filesystem::path output_path = "data/gemmv3.dat";
        std::ofstream output_file(output_path, std::ios::binary);
        output_file.write(reinterpret_cast<char *>(c), n1 * n3 * sizeof(float));        



        cudaFree(d_a); cudaFree(d_b); cudaFree(d_c);
    }
    else{
        std::cerr << "Invalid version: " << version << std::endl;
        return -1;
    }

    delete [] a;
    delete [] b;
    delete [] c;




    return 0;
}