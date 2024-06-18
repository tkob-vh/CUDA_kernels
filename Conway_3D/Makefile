all: generate baseline

generate: generate.cpp
					g++ -O3 generate.cpp -o generate -std=c++17

baseline: baseline.cpp
					g++ -O3 baseline.cpp -o baseline -std=c++17

answer: answer.cu
				nvcc -O3 answer.cu -o answer --ptxas-options=-v -gencode arch=compute_89,code=sm_89

official: official.cu
				nvcc -O3 official.cu -o official --ptxas-options=-v -gencode arch=compute_89,code=sm_89

.PHONY: clean
clean: 
			rm generate baseline