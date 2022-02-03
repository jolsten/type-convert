import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u8)',
]


def func(value: np.uint64) -> np.float64:
    value = np.uint64(value)

    s = (value >> np.uint8(63)) & np.uint64(1)
    e = (value >> np.uint8(55)) & np.uint64(0xFF)
    m = (value & np.uint64(0x007FFFFFFFFFFFFF))

    S = np.int8(-1) ** s
    E = np.int16(e) - np.int16(128)
    M = np.float64(m) / np.float64(2**56) + np.float64(0.5)

    return np.float64(S * M * np.float64(2)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
