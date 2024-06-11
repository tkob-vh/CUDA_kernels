import numpy as np
import struct
import argparse
import random
import string

def generate_input_data(length):
    letters = string.ascii_letters #+ string.digits + string.punctuation
    in_data = ''.join(random.choice(letters) for _ in range(length))

    with open("data/input.dat", "wb") as fi:
        fi.write(struct.pack("i", length))
        fi.write(in_data.encode('utf-8'))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=int, default=102400, help="Length of the random letters")

    args = parser.parse_args()

    generate_input_data(args.length)