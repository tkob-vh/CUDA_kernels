import struct
import numpy as np

def generate_conf_data():
    n1, n2, n3 = 8192, 8192, 8192

    # Generate random data
    a = np.random.rand(n1, n2).astype(np.float32)
    b = np.random.rand(n2, n3).astype(np.float32)


    with open("data/input.dat", "wb") as fi:
        # 写入 n1、n2、n3 到文件
        fi.write(struct.pack("i", n1))
        fi.write(struct.pack("i", n2))
        fi.write(struct.pack("i", n3))

        # 写入数组数据 a 和 b 到文件
        fi.write(a.tobytes())
        fi.write(b.tobytes())

if __name__ == "__main__":
    generate_conf_data()