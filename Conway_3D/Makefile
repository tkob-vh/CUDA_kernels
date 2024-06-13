all: generate baseline

generate: generate.cpp
					g++ -O3 generate.cpp -o generate -std=c++17

baseline: baseline.cpp
					g++ -O3 baseline.cpp -o baseline -std=c++17

.PHONY: clean
clean: 
			rm generate baseline