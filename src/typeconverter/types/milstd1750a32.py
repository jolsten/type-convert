import numpy as np
from numba import njit, vectorize
from .twoscomp import jfunc as uint_to_twoscomp

signatures = [
    'f4(u4)',
]

# Reference(s):
# http://www.xgc-tek.com/manuals/mil-std-1750a/c191.html


def func(value: np.uint32) -> np.float32:
    value = np.uint32(value)
    m = uint_to_twoscomp(
        (value & np.uint32(0xFFFFFF00)) >> np.uint8(8), np.uint8(24)
    )
    e = uint_to_twoscomp(value & np.uint32(0x000000FF), np.uint8(8))
    M = np.float32(m) / np.float32(2**23)
    E = np.float32(e)
    return np.float32(M * 2 ** E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
