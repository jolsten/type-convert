import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u8)',
]


def func(value: np.uint64) -> np.float64:
    value = np.uint64(value)

    s = (value >> np.uint8(63)) & np.uint64(1)
    e = (value >> np.uint8(52)) & np.uint64(0x7FF)
    m = (value & np.uint64(0x000FFFFFFFFFFFFF))

    print('A', s, e, m)

    S = np.int8(-1) ** s
    E = np.int16(e) - np.int16(1024)
    M = np.float64(m) / np.float64(2**53) + np.float64(0.5)

    print('B', S, E, M)

    return np.float64(S * M * np.float64(2)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
