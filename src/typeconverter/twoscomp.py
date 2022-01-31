import numpy as np
from numba import njit, vectorize

from .types import UnsignedInteger, SignedInteger

signatures = [
    'i1(u1,u1)',
    'i2(u2,u1)',
    'i4(u4,u1)',
    'i8(u8,u1)',
]

# def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
#     value = np.uint64(value)
#     if value & (np.uint64(1) << np.uint8(size-1)) != 0:
#         pad_bits = np.uint8(64-size)
#         value = -np.int64((~(value << pad_bits)) >> pad_bits) - 1
#     return value

# def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
#     value = np.uint64(value)
#     pad_bits = np.uint8(64-size)
#     if value & (np.uint64(1) << np.uint8(size-1)) != 0:
#         return -np.int64((~(value << pad_bits)) >> pad_bits) - 1
#     return value

def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
    value = np.uint64(value)
    if value >= 2**(size-1):
        pad_bits = np.uint8(64 - size)
        value = np.int64(np.uint64(value) << pad_bits) >> pad_bits
    return value


# def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
#     # print('A', type(value), value, type(size), size)
#     if (value & (1 << (size - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
#         x = np.uint64(1) << np.uint8(size)
#         # print('x', type(x), x)
#         # print('y', type(value), value)
#         value = np.int64(value - x)      # compute negative value
#     # print('B', type(value), value, type(size), size)
#     return value                         # return positive value as is

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
