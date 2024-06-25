#include <iostream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 16
#define INPUT_TILE BLOCK_WIDTH
#define OUTPUT_TILE 14
#define Z_ITER 14

namespace fs = std::filesystem;

__global__ void conway_step(uint8_t *curr_space, uint8_t *next_space, size_t M) {
  __shared__ uint8_t curr_space_fro_s[INPUT_TILE][INPUT_TILE];
  __shared__ uint8_t curr_space_mid_s[INPUT_TILE][INPUT_TILE];
  __shared__ uint8_t curr_space_end_s[INPUT_TILE][INPUT_TILE];

  int tx = threadIdx.x;
  int ty = threadIdx.y;
  
  // Initialize the shared memory
  curr_space_fro_s[ty][tx] = 0;
  curr_space_mid_s[ty][tx] = 0;
  curr_space_end_s[ty][tx] = 0;

  int iStart = Z_ITER * blockIdx.z; // the start index of the output_tile in z axis.
  int j = OUTPUT_TILE * blockIdx.y + ty - 1;
  int k = OUTPUT_TILE * blockIdx.x + tx - 1;

  int mj_k = ((j + M) % M) * M + (k + M) % M; 
  int M2 = M * M;

  curr_space_fro_s[ty][tx] = curr_space[((iStart + M - 1) % M) * M2 + mj_k];
  curr_space_mid_s[ty][tx] = curr_space[iStart * M2 + mj_k];

  for(int i = iStart; i < iStart + Z_ITER; i++) {
    curr_space_end_s[ty][tx] = curr_space[((i + 1) % M) * M2 + mj_k];

    __syncthreads();

    if((unsigned int)(i - 0) < M && (unsigned int)(j - 0) < M && (unsigned int)(k - 0) < M && 
       (unsigned int)(tx - 1) < INPUT_TILE - 2 && (unsigned int)(ty - 1) < INPUT_TILE - 2 ) {
        uint8_t neighbor_count = curr_space_mid_s[ty - 1][tx - 1] + curr_space_mid_s[ty - 1][tx]
                               + curr_space_mid_s[ty - 1][tx + 1] + curr_space_mid_s[ty][tx - 1]
                               + curr_space_mid_s[ty][tx + 1] + curr_space_mid_s[ty + 1][tx - 1]
                               + curr_space_mid_s[ty + 1][tx] + curr_space_mid_s[ty + 1][tx + 1]
                               + curr_space_fro_s[ty - 1][tx - 1] + curr_space_fro_s[ty - 1][tx]
                               + curr_space_fro_s[ty - 1][tx + 1] + curr_space_fro_s[ty][tx - 1]
                               + curr_space_fro_s[ty][tx] + curr_space_fro_s[ty][tx + 1]
                               + curr_space_fro_s[ty + 1][tx - 1] + curr_space_fro_s[ty + 1][tx]
                               + curr_space_fro_s[ty + 1][tx + 1] + curr_space_end_s[ty - 1][tx - 1]
                               + curr_space_end_s[ty - 1][tx] + curr_space_end_s[ty - 1][tx + 1]
                               + curr_space_end_s[ty][tx - 1] + curr_space_end_s[ty][tx]
                               + curr_space_end_s[ty][tx + 1] + curr_space_end_s[ty + 1][tx - 1]
                               + curr_space_end_s[ty + 1][tx] + curr_space_end_s[ty + 1][tx + 1];

        uint8_t curr_state = curr_space_mid_s[ty][tx];
        uint8_t &next_state = next_space[i * M2 + mj_k];  // Corrected to use correct indexing                        

        if(curr_state == 1) {
          if(neighbor_count < 5 || neighbor_count > 7) next_state = 0;
          else next_state = 1;
        } else {
          if(neighbor_count == 6) next_state = 1;
          else next_state = 0;
        }

    }

    __syncthreads();

    curr_space_fro_s[ty][tx] = curr_space_mid_s[ty][tx];
    curr_space_mid_s[ty][tx] = curr_space_end_s[ty][tx];
  }
}

int main(int argc, char *argv[]) {
  if(argc < 4) {
    std::cout << "Usage: " << argv[0] << " <input_path> <output_path> <N>"
              << std::endl;
    return 1;
  }

  fs::path input_path = argv[1];
  fs::path output_path = argv[2];

  int N = std::atoi(argv[3]);

  size_t M, T;

  std::ifstream input_file(input_path, std::ios::binary);
  input_file.read(reinterpret_cast<char *>(&M), sizeof(M));
  input_file.read(reinterpret_cast<char *>(&T), sizeof(T));

  size_t M3 = M * M * M;

  uint8_t *curr_space = new uint8_t[M3];
  uint8_t *next_space = new uint8_t[M3];

  input_file.read(reinterpret_cast<char *>(curr_space), M3);

  uint8_t *curr_space_d, *next_space_d;
  cudaMalloc(&curr_space_d, M3 * sizeof(uint8_t));
  cudaMalloc(&next_space_d, M3 * sizeof(uint8_t));
  cudaMemcpy(curr_space_d, curr_space, M3 * sizeof(uint8_t), cudaMemcpyHostToDevice);

  dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  dim3 gridDim(ceil((float) M / OUTPUT_TILE), ceil((float) M / OUTPUT_TILE), ceil((float) M / Z_ITER));

  auto t1 = std::chrono::steady_clock::now();

  for(int i = 0; i < N / 2; i++) {
    conway_step<<<gridDim, blockDim>>>(curr_space_d, next_space_d, M);
    conway_step<<<gridDim, blockDim>>>(next_space_d, curr_space_d, M);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
      std::cerr << "CUDA kernel launched failed: " << cudaGetErrorString(error) << std::endl;
    }
  }

  cudaDeviceSynchronize();
  auto t2 = std::chrono::steady_clock::now();
  cudaMemcpy(curr_space, curr_space_d, M3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  
  T += N;
  std::ofstream output_file(output_path, std::ios::binary);
  output_file.write(reinterpret_cast<char *>(&M), sizeof(M));
  output_file.write(reinterpret_cast<char *>(&T), sizeof(T));
  output_file.write(reinterpret_cast<char *>(curr_space), M3 * sizeof(uint8_t));

  delete[] curr_space;
  delete[] next_space;

  cudaFree(curr_space_d);
  cudaFree(next_space_d);

  int duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << duration << std::endl;

  return 0;
}
