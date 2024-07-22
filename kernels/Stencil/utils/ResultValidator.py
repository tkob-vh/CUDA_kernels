import struct
import numpy as np
import argparse

def read_input_file(filename):
    with open(filename, "rb") as fi:
        nx = struct.unpack("i", fi.read(4))[0]
        ny = struct.unpack("i", fi.read(4))[0]
        nz = struct.unpack("i", fi.read(4))[0]
    return nx, ny, nz

def read_result(filename, nx, ny, nz):
    with open(filename, "rb") as fi:
        c = np.frombuffer(fi.read(nx * ny * nz * 4), dtype=np.float32).reshape(nz, ny, nx)
    return c

def calculate_error(c1, c2):
    mse = np.mean((c1 - c2) ** 2)
    return mse

def main(test_file):
    nx, ny, nz = read_input_file("data/input.dat")

    c_ref = read_result("data/stencil_v0.dat", nx, ny, nz)
    c_test = read_result(test_file, nx, ny, nz)

    error = calculate_error(c_ref, c_test)
    print(f"Mean Squared Error between calculated and reference results: {error}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the result of GEMM")
    parser.add_argument("test_file", help="The file containing the result of the test")
    args = parser.parse_args()

    main(args.test_file)
