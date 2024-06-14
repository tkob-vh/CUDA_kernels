#include <iostream>
#include <string>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <utility>
#include <cuda_runtime.h>

namespace fs = std::filesystem;

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

  uint8_t *curr_space = new uint8_t[M * M * M];
  uint8_t *next_space = new uint8_t[M * M * M];

  input_file.read(reinterpret_cast<char *>(curr_space), M * M * M);

  uint8_t *curr_space_d, *next_space_d;
  cudaMalloc(&curr_space_d, M * M * M * sizeof(uint8_t));
  cudaMalloc(&next_space_d, M * M * M * sizeof(uint8_t));
  cudaMemcpy(curr_space_d, curr_space, M * M * M * sizeof(uint8_t), cudaMemcpyHostToDevice);

  // cuda kernel

  cudaMemcpy(curr_space, curr_space_d, M * M * M * sizeof(uint8_t), cudaMemcpyDeviceToHost);

  T += N;
  std::ofstream output_file(output_path, std::ios::binary);
  output_file.write(reinterpret_cast<char *>(&M), sizeof(M));
  output_file.write(reinterpret_cast<char *>(&T), sizeof(T));
  output_file.write(reinterpret_cast<char *>(curr_space), M * M * M * sizeof(uint8_t));

  delete[] curr_space;
  delete[] next_space;

  cudaFree(curr_space_d);
  cudaFree(next_space_d);

  return 0;
}