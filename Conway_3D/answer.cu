#include <iostream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

#define Z_WIDTH 4

namespace fs = std::filesystem;


__global__ void conway_step(uint8_t *curr_space, uint8_t *next_space, size_t width, size_t M) {
  extern __shared__ char inputs[];
  uint32_t *src = (uint32_t *)inputs; // One uint32_t stores 4 cells in the x axis.
  uint32_t *input = (uint32_t *)(inputs + width * Z_WIDTH * sizeof(uint32_t));
  uint8_t *src_ = (uint8_t*)src;

  int y = blockIdx.y;
  int zStart = blockIdx.z; // One thread is responsible for Z_WIDTH cells starting at zStart in the z axis.
  int tx = threadIdx.x;

  // Calculate the y index needed for the 4 cells in one thread.
  int y_index[3];
  y_index[0] = (y + M - 1) % M;
  y_index[1] = y;
  y_index[2] = (y + 1) % M;

  uint32_t count[Z_WIDTH + 2]; // Store the number of alive cells in the 3 * 4 cells near the target cell.

  for(int i = 0; i < Z_WIDTH + 2; i++) { // Iterate through z axis.
    int zCurr = (zStart * Z_WIDTH + i - 1 + M) % M;

    count[i] = 0;
    for(int j = 0; j < 3; j++) { // Iterate through y axis.
      count[i] += curr_space[(zCurr * M + y_index[j]) * width + tx];
    }
  }

  // Load the input cell into shared memory
  for(int i = 0; i < Z_WIDTH; i++) {
    src[i * width + tx] = curr_space[(zStart * Z_WIDTH + i) * width * M + y * width + tx];
  }

  __syncthreads();
  // Calculate the number of alive cells across 3 z indexs. Finally we get the number of alive cells for each Z_WIDTH cell without considering its x-axis neighbors.
  for(int i = 0; i < Z_WIDTH; i++) {
    count[i] = count[i] + count[i + 1] + count[i + 2];
  }

  // Load the count array from local memory to shared memory.
  for(int i = 0; i < Z_WIDTH; i++) {
    input[i * (width + 2) + tx + 1] = count[i];
  }

  __syncthreads();

  // Load the count array for the cells in the near the halo.
  if(tx < Z_WIDTH) {
    input[tx * (width + 2)] = input[tx * (width + 2) + width];
    input[tx * (width + 2) + (width + 1)] = input[tx * (width + 2) + 1];
  }

  __syncthreads();

  uint32_t tempout[4];
  for(int i = 0; i < Z_WIDTH; i++) {
    // 
    uint32_t left = input[i * (width + 2) + tx];
    uint32_t center = input[i * (width + 2) + tx + 1];
    uint32_t right = input[i * (width + 2) + tx + 2];
    center = center + (center >> 8) + (right << 24) + (center << 8) + (left >> 24);

    for(int j = 0; j < 4; j++) {
      uint32_t c = center & 0xff;
      center = center >> 8;

      if(c == 6 || 6 <= c && c <= 8 && src_[(i * width + tx) * 4 + j]) {
        tempout[j] = 1;
      }
      else {
        tempout[j] = 0;
      }
    }

    uint32_t out = tempout[0] + (tempout[1] << 8) + (tempout[2] << 16) + (tempout[3] << 24);
    next_space[((zStart * Z_WIDTH + i) * M + y) * width + tx] = out;

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

  cudaError_t err;
  uint8_t *curr_space_d, *next_space_d;
  err = cudaMalloc(&curr_space_d, M3 * sizeof(uint8_t));
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  err = cudaMalloc(&next_space_d, M3 * sizeof(uint8_t));
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  err = cudaMemcpy(curr_space_d, curr_space, M3 * sizeof(uint8_t), cudaMemcpyHostToDevice);
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  auto t1 = std::chrono::steady_clock::now();

  size_t width = M / 4; 
  dim3 blockDim(width, 1, 1);
  dim3 gridDim(1, M, M / Z_WIDTH);

  size_t shared_region = width *  Z_WIDTH * sizeof(uint32_t) + (width + 2) * (Z_WIDTH + 2) * sizeof(uint32_t);

  for(int i = 0; i < N / 2; i++) {
    conway_step<<<gridDim, blockDim, shared_region>>>(curr_space_d, next_space_d, width, M);

    conway_step<<<gridDim, blockDim, shared_region>>>(next_space_d, curr_space_d, width, M);
  }
  err = cudaGetLastError();
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  

  err = cudaDeviceSynchronize();
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }

  auto t2 = std::chrono::steady_clock::now();

  err = cudaMemcpy(curr_space, curr_space_d, M3 * sizeof(uint8_t), cudaMemcpyDeviceToHost);
  if(err != cudaSuccess) {
    std::cerr << cudaGetErrorString(err) << std::endl;
  }
  
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
