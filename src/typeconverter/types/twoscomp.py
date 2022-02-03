import numpy as np
from numba import njit, vectorize
from typeconverter.typing import UnsignedInteger, SignedInteger

signatures = [
    'i1(u1,u1)',
    'i2(u2,u1)',
    'i4(u4,u1)',
    'i8(u8,u1)',
]


def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
    value = np.uint64(value)
    if value >= 2**(size-1):
        pad_bits = np.uint8(64 - size)
        value = np.int64(np.uint64(value) << pad_bits) >> pad_bits
    return value


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
