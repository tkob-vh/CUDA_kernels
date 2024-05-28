#include<iostream>
#include<cstdlib>
#include<chrono>
#include<string>
#include<mkl.h>
#include<omp.h>
#include<cuda_runtime.h>
#include"gemm.hh"

int main(int argc, char **argv){
    // Check the number of arguments
    if(argc != 2){
        std::cerr << "Usage: " << argv[0] << " (cpu|gemmv0|gemmv1|gemmv2)" << std::endl;
        return 1;
    }

    std::string mode = argv[1]; 

    uint32_t n1, n2, n3;
    FILE *fi;

    // Read the input data
    printf("Begin to read the input data\n");
    fi = fopen("data/input.dat", "rb");
    fread(&n1, 1, sizeof(int), fi);
    fread(&n2, 1, sizeof(int), fi);
    fread(&n3, 1, sizeof(int), fi);
    printf("n1: %u, n2: %u, n3: %u\n", n1, n2, n3);

    float *a = (float *)malloc(n1 * n2 * sizeof(float));
    float *b = (float *)malloc(n2 * n3 * sizeof(float));
    float *c = (float *)malloc(n1 * n3 * sizeof(float));

    fread(a, 1, n1 * n2 * sizeof(float), fi);
    fread(b, 1, n2 * n3 * sizeof(float), fi);
    fclose(fi);
    printf("The input data has been read\n");

    if(mode == "cpu"){
        printf("The mode is cpu\n");
        // Initialize c

        // Measure the time
        auto t1 = std::chrono::steady_clock::now();
        cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n1, n3, n2, 1.0, a, n2, b, n3, 0.0, c, n3);
        auto t2 = std::chrono::steady_clock::now();
        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
        printf("%d\n", d1);

        // Write the result
        fi = fopen("data/ref.dat", "wb");
        fwrite(c, 1, n1 * n3 * sizeof(float), fi);
        fclose(fi);

        free(a);
        free(b);
        free(c);

    }
    else if(mode == "gemmv0"){
        printf("The mode is gemmv0\n");
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim(ceil((float)n3 / blockDim.x), ceil((float)n1 / blockDim.y));

        // Measure the time
        auto t1 = std::chrono::steady_clock::now();
        matrixMul0<<<gridDim, blockDim>>>(d_a, d_b, d_c, n1);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);

        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        printf("%d\n", d1);
        fi = fopen("data/gemmv0.dat", "wb");
        fwrite(c, 1, n1 * n3 * sizeof(float), fi);
        fclose(fi);

        free(a);
        free(b);
        free(c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    else if(mode == "gemmv1"){
        printf("The mode is gemmv1\n");
        float *d_a, *d_b, *d_c;
        cudaMalloc(&d_a, n1 * n2 * sizeof(float));
        cudaMalloc(&d_b, n2 * n3 * sizeof(float));
        cudaMalloc(&d_c, n1 * n3 * sizeof(float));

        cudaMemcpy(d_a, a, n1 * n2 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(d_b, b, n2 * n3 * sizeof(float), cudaMemcpyHostToDevice);

        dim3 blockDim(TILE_WIDTH, TILE_WIDTH);
        dim3 gridDim(ceil((float)n3 / blockDim.x), ceil((float)n1 / blockDim.y));

        // Measure the time
        auto t1 = std::chrono::steady_clock::now();
        matrixMul1<<<gridDim, blockDim>>>(d_a, d_b, d_c, n1, n2, n3);
        cudaDeviceSynchronize();
        auto t2 = std::chrono::steady_clock::now();
        cudaMemcpy(c, d_c, n1 * n3 * sizeof(float), cudaMemcpyDeviceToHost);

        int d1 = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();

        printf("%d\n", d1);
        fi = fopen("data/gemmv1.dat", "wb");
        fwrite(c, 1, n1 * n3 * sizeof(float), fi);
        fclose(fi);

        free(a);
        free(b);
        free(c);
        cudaFree(d_a);
        cudaFree(d_b);
        cudaFree(d_c);
    }
    else{
        std::cerr << "Invalid mode: " << mode << std::endl;
        return 1;
    }





    return 0;
}