import taichi as ti
import numpy as np

from utils import T, scalars, get_reduce_min_inplace


if __name__ == "__main__":
    ti.init(arch=ti.gpu, debug=True)

    n = 5
    keys = scalars(T, n)
    values = scalars(ti.i32, n)
    keys.from_numpy(np.array([2, 4, 3, 1, 6], dtype=np.float32))
    values.from_numpy(np.arange(n, dtype=np.int32))

    reduce_min_func, count = get_reduce_min_inplace(keys, values)

    @ti.kernel
    def perform_reduce():
        print("begin reduce")
        for i in ti.ndrange(count // 2):
            reduce_min_func(i)

    perform_reduce()

    print("result:", keys[0], values[0])
