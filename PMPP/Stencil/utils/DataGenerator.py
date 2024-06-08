import numpy as np
import struct
import argparse

def generate_input_data(nx, ny, nz):
    in_dat = np.random.rand(nx, ny, nz).astype(np.float32)

    with open("data/input.dat", "wb") as fi:
        fi.write(struct.pack("i", nx))
        fi.write(struct.pack("i", ny))
        fi.write(struct.pack("i", nz))
        fi.write(in_dat.tobytes())

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nx", type=int, default=512, help="The number of elements in the x axis")
    parser.add_argument("--ny", type=int, default=512, help="The number of elements in the y axis")
    parser.add_argument("--nz", type=int, default=512, help="The number of elements in the z axis")

    args = parser.parse_args()

    generate_input_data(args.nx, args.ny, args.nz)