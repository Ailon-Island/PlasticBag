import taichi as ti
import numpy as np

from utils import T, scalars


if __name__ == "__main__":
    ti.init(arch=ti.gpu, debug=True)

    n = 5
    count = int(2 ** np.ceil(np.log2(n)))
    keys = scalars(T, n)
    values = scalars(ti.i32, n)
    keys.from_numpy(np.array([2, 4, 3, 1, 6], dtype=np.float32))
    values.from_numpy(np.arange(n, dtype=np.int32))

    @ti.func
    def reduce_min_func(i):
        s = count // 2
        while s > 0:
            if i == 0:
                print(s)
            if i < s and i + s < n:
                i_small = keys[i] < keys[i + s]
                keys[i] = keys[i] if i_small else keys[i + s]
                values[i] = values[i] if i_small else values[i + s]
                s >>= 1
            ti.sync()
            print(i, keys[i], values[i])
        

    @ti.kernel
    def perform_reduce():
        print("begin reduce")
        for i in ti.ndrange(count // 2):
            reduce_min_func(i)

    perform_reduce()

    print(keys[0], values[0])
