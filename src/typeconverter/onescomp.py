import numpy as np
from numba import njit, vectorize, generated_jit

from .typing import UnsignedInteger, SignedInteger

signatures = [
    'i1(u1,u1)',
    'i2(u2,u1)',
    'i4(u4,u1)',
    'i8(u8,u1)',
]

def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
    value = np.uint64(value)
    pad_bits = np.uint8(64-size)
    if value & (np.uint64(1) << np.uint8(size-1)) != 0:
        return -np.int64((~(value << pad_bits)) >> pad_bits)
    return value

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
