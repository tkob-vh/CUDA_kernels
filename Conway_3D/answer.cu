#include <iostream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

__global__ void conway_step(uint8_t *curr_space, uint8_t *next_space, size_t M) {
  int x = blockDim.x * blockIdx.x + threadIdx.x;
  int y = blockDim.y * blockIdx.y + threadIdx.y;
  int z = blockDim.z * blockIdx.z + threadIdx.z;

  int lx = (x + M - 1) % M * M * M;
  int mx = x * M * M;
  int rx = (x + 1) % M * M * M;
  int ly = (y + M - 1) % M * M;
  int my = y * M;
  int ry = (y + 1) % M * M;
  int lz = (z + M - 1) % M;
  int mz = z;
  int rz = (z + 1) % M;

  uint8_t curr_state = curr_space[mx + my + mz];
  uint8_t &next_state = next_space[mx + my + mz];

  uint8_t neighbor_count = curr_space[lx + ly + lz] + curr_space[lx + ly + mz] +
                          curr_space[lx + ly + rz] + curr_space[lx + my + lz] +
                          curr_space[lx + my + mz] + curr_space[lx + my + rz] +
                          curr_space[lx + ry + lz] + curr_space[lx + ry + mz] +
                          curr_space[lx + ry + rz] + curr_space[mx + ly + lz] +
                          curr_space[mx + ly + mz] + curr_space[mx + ly + rz] +
                          curr_space[mx + my + lz] + curr_space[mx + my + rz] +
                          curr_space[mx + ry + lz] + curr_space[mx + ry + mz] +
                          curr_space[mx + ry + rz] + curr_space[rx + ly + lz] +
                          curr_space[rx + ly + mz] + curr_space[rx + ly + rz] +
                          curr_space[rx + my + lz] + curr_space[rx + my + mz] +
                          curr_space[rx + my + rz] + curr_space[rx + ry + lz] +
                          curr_space[rx + ry + mz] + curr_space[rx + ry + rz];

  if(curr_state == 1) {
    if(neighbor_count < 5 || neighbor_count > 7) next_state = 0;
    else next_state = 1;
  }
  else {
    if(neighbor_count == 6) next_state = 1;
    else next_state = 0;
  }
}

int main(int argc, char *argv[]) {
  if(argc < 4) {
    std::cout << "Usage: " << argv[0] << " <input_path> <output_path> <N>"
              << std::endl;
    return 1;
  }
  auto t1 = std::chrono::steady_clock::now();

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

  dim3 blockDim(16, 8, 8);
  dim3 gridDim(ceil((float) M / blockDim.x), ceil((float) M / blockDim.y), ceil((float) M / blockDim.z));

  for(int i = 0; i < N; i++){
    conway_step<<<gridDim, blockDim>>>(curr_space_d, next_space_d, M);
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
      std::cerr << "CUDA kernel launched failed: " << cudaGetErrorString(error) << std::endl;
    }
    cudaDeviceSynchronize();
    std::swap(curr_space_d, next_space_d);
  }

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

  auto t2 = std::chrono::steady_clock::now();
  int duration = std::chrono::duration_cast<std::chrono::milliseconds>(t2 - t1).count();
  std::cout << duration << std::endl;

  return 0;
}