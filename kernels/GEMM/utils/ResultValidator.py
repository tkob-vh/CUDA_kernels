import struct
import numpy as np
import argparse

def read_input_file(filename):
    with open(filename, "rb") as f:
        n1 = struct.unpack("i", f.read(4))[0]
        n2 = struct.unpack("i", f.read(4))[0]
        n3 = struct.unpack("i", f.read(4))[0]
        
    return n1, n2, n3

def read_result(filename, n1, n3):
    with open(filename, "rb") as f:
        c = np.frombuffer(f.read(n1 * n3 * 4), dtype=np.float32).reshape(n1, n3)
    return c

def calculate_error(c1, c2):
    mse = np.mean((c1 - c2) ** 2)
    return mse

def main(test_file):

    n1, n2, n3 = read_input_file("data/input.dat")

    c_ref = read_result("data/ref.dat", n1, n3)
    c_test = read_result(test_file, n1, n3)

    error = calculate_error(c_ref, c_test)
    print(f"Mean Squared Error between calculated and reference results: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the result of GEMM")
    parser.add_argument("test_file", help="The file containing the result of the test")
    args = parser.parse_args()

    main(args.test_file)
