import struct
import numpy as np
import argparse

def read_input_file(filename):
    with open(filename, "rb") as f:
        width = struct.unpack("i", f.read(4))[0]
        height = struct.unpack("i", f.read(4))[0]
        r = struct.unpack("i", f.read(4))[0]
    return width, height, r

def read_result(filename, width, height):
    with open(filename, "rb") as f:
        c = np.frombuffer(f.read(width * height * 4), dtype=np.float32).reshape(width, height)
    return c

def calculate_error(c1, c2):
    mse = np.mean((c1 - c2) ** 2)
    return mse

def main(test_file):

    width, height, r = read_input_file("data/input.dat")

    c_ref = read_result("data/ref.dat", width, height)
    c_test = read_result(test_file, width, height)

    error = calculate_error(c_ref, c_test)
    print(f"Mean Squared Error between calculated and reference results: {error}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate the result of Convolution")
    parser.add_argument("test_file", help="The file containing the result of the test")
    args = parser.parse_args()

    main(args.test_file)