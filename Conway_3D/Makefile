all: generate baseline answer_v0 answer_v1 answer_v2 answer_v3

generate: generate.cpp
					g++ -O3 generate.cpp -o generate -std=c++17

baseline: baseline.cpp
					g++ -O3 baseline.cpp -o baseline -std=c++17

answer_v0: answer_v0.cu
				nvcc -O3 answer_v0.cu -o answer --ptxas-options=-v -gencode arch=compute_89,code=sm_89

answer_v1: answer_v1.cu
				nvcc -O3 answer_v1.cu -o answer --ptxas-options=-v -gencode arch=compute_89,code=sm_89

answer_v2: answer_v2.cu
				nvcc -O3 answer_v2.cu -o answer --ptxas-options=-v -gencode arch=compute_89,code=sm_89

answer_v3: answer_v3.cu
				nvcc -O3 answer_v3.cu -o answer --ptxas-options=-v -gencode arch=compute_89,code=sm_89


.PHONY: clean
clean: 
			rm generate baseline answer_v0 answer_v1 answer_v2 answer_v3