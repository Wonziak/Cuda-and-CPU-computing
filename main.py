import numpy as np
from timeit import default_timer as timer
from numba import vectorize, cuda, njit, prange
import numba.cuda


@vectorize(["float32(float32, float32)"], target='cuda')
def vector_add(a, b):
    return a+b


def vector_add2(a, b, c):
    for i in range(a.size):
        c[i] = a[i] + b[i]


@vectorize(["float32(float32, float32)"], target='cpu')
def vector_add3(a, b):
    return a + b


def main():
    n = 60000000
    a = np.ones(n, dtype=np.float32)

    start = timer()
    vector_add(a, a)
    vector_add_time = timer() - start
    print("Time gpu " + str(vector_add_time))

    start = timer()
    vector_add(a, a)
    vector_add_time = timer() - start
    print("Time gpu " + str(vector_add_time))

    start = timer()
    vector_add3(a, a)
    vector_add_time3 = timer() - start
    print("Time cpu " + str(vector_add_time3))

    start = timer()
    vector_add3(a, a)
    vector_add_time3 = timer() - start
    print("Time cpu " + str(vector_add_time3))


if __name__ == '__main__':
    main()
