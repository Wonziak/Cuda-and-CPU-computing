import numpy as np
import time
from numba import njit, prange, cuda
import timeit
import numba.cuda
import math


@numba.cuda.jit
def smooth_gpu(x, out):
    j, i = cuda.grid(2)
    m, n = x.shape

    if 1 <= i < n - 1 and 1 <= j < m - 1:
        out[i, j] = (x[i - 1, j - 1] + x[i - 1, j] + x[i - 1, j + 1] +
                     x[i, j - 1] + x[i, j] + x[i, j + 1] +
                     x[i + 1, j - 1] + x[i + 1, j] + x[i + 1, j + 1]) / 9


x = np.ones((10000, 10000), dtype='int8')
out = np.zeros((10000, 10000), dtype='int8')
x_gpu = cuda.to_device(x)
out_gpu = cuda.device_array_like(out)
threadsperblock = (16, 16)
blockspergrid_x = math.ceil(x_gpu.shape[0] / threadsperblock[0])
blockspergrid_y = math.ceil(x_gpu.shape[1] / threadsperblock[1])
blockspergrid = (blockspergrid_x, blockspergrid_y)

# run on gpu
smooth_gpu[blockspergrid, threadsperblock](x_gpu, out_gpu)  # compile before measuring time
cuda.synchronize()
start_time = time.time()
smooth_gpu[blockspergrid, threadsperblock](x_gpu, out_gpu)
cuda.synchronize()
print("GPU Time: " + str(time.time() - start_time))

x_cpu = np.ones((10000, 10000), dtype='int8')
out_cpu = np.zeros((10000, 10000), dtype='int8')


@njit(parallel=True)
def smooth_cpu(x, out_cpu):
    for i in prange(1, x.shape[0] - 1):
        for j in range(1, x.shape[1] - 1):
            out_cpu[i, j] = (x[i - 1, j - 1] + x[i - 1, j] + x[i - 1, j + 1] + x[i, j - 1] + x[i, j] + x[i, j + 1] + x[
                i + 1, j - 1] + x[i + 1, j] + x[i + 1, j + 1]) / 9


# run on cpu
smooth_cpu(x_cpu, out_cpu)  # compile before measuring time
start_time = time.time()
smooth_cpu(x_cpu, out_cpu)
print("CPU Time: {0:1.6f}s ".format(time.time() - start_time))
