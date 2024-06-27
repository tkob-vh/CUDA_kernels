#include <iostream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <chrono>
#include <cuda_runtime.h>

#define BLOCK_WIDTH 32
#define Z_ITER 32
namespace fs = std::filesystem;

__global__ void conway_step(uint8_t *curr_space, uint8_t *next_space, size_t M) {

  int i = blockDim.z * blockIdx.z + threadIdx.z;
  int j = blockDim.y * blockIdx.y + threadIdx.y;
  int k = blockDim.x * blockIdx.x + threadIdx.x;

  int li = (i + M - 1) % M * M * M;
  int mi = i * M * M;
  int ri = (i + 1) % M * M * M;

  int lj = (j + M - 1) % M * M;
  int mj = j * M;
  int rj = (j + 1) % M * M;

  int lk = (k + M - 1) % M;
  int mk = k;
  int rk = (k + 1) % M;

  uint8_t curr_state = curr_space[mi + mj + mk];
  uint8_t &next_state = next_space[mi + mj + mk];

  uint8_t neighbor_count = curr_space[li + lj + lk] + curr_space[li + lj + mk] +
                          curr_space[li + lj + rk] + curr_space[li + mj + lk] +
                          curr_space[li + mj + mk] + curr_space[li + mj + rk] +
                          curr_space[li + rj + lk] + curr_space[li + rj + mk] +
                          curr_space[li + rj + rk] + curr_space[mi + lj + lk] +
                          curr_space[mi + lj + mk] + curr_space[mi + lj + rk] +
                          curr_space[mi + mj + lk] + curr_space[mi + mj + rk] +
                          curr_space[mi + rj + lk] + curr_space[mi + rj + mk] +
                          curr_space[mi + rj + rk] + curr_space[ri + lj + lk] +
                          curr_space[ri + lj + mk] + curr_space[ri + lj + rk] +
                          curr_space[ri + mj + lk] + curr_space[ri + mj + mk] +
                          curr_space[ri + mj + rk] + curr_space[ri + rj + lk] +
                          curr_space[ri + rj + mk] + curr_space[ri + rj + rk];

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
  // dim3 blockDim(BLOCK_WIDTH, BLOCK_WIDTH, 1);
  // dim3 gridDim(ceil((float) M / blockDim.x), ceil((float) M / blockDim.y), ceil((float) M / Z_ITER));


  auto t1 = std::chrono::steady_clock::now();

  for(int i = 0; i < N; i++){
    conway_step<<<gridDim, blockDim>>>(curr_space_d, next_space_d, M);
    #ifdef DEBUG
    cudaError_t error = cudaGetLastError();
    if(error != cudaSuccess) {
      std::cerr << "CUDA kernel launched failed: " << cudaGetErrorString(error) << std::endl;
    }
    #endif
    cudaDeviceSynchronize();
    std::swap(curr_space_d, next_space_d);
  }

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