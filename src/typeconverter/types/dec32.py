import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u4)',
]


def func(value: np.uint32) -> np.float64:
    value = np.uint32(value)

    s = (value & np.uint32(0x80000000)) >> np.uint8(31)
    e = (value & np.uint32(0x7F800000)) >> np.uint8(23)
    m = (value & np.uint32(0x007FFFFF))

    S = np.int8(-1) ** s
    E = np.int16(e) - np.int16(128)
    M = np.float64(m) / np.float64(2**24)

    return np.float64(S * M * np.float64(2)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
