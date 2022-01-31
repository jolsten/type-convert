import numpy as np
from numba import njit, vectorize, generated_jit

from .types import UnsignedInteger, SignedInteger

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

    # if size == 8:
    #     value = np.uint8(value).view(np.int8)
    #     if value < 0:
    #         value += np.int8(1)
    #     return value
    # elif size == 16:
    #     value = np.uint16(value).view(np.int16)
    #     if value < 0:
    #         value += np.int16(1)
    #     return value
    # elif size == 32:
    #     value = np.uint32(value).view(np.int32)
    #     if value < 0:
    #         value += np.int32(1)
    #     return value
    # elif size == 64:
    #     value = np.uint64(value).view(np.int64)
    #     if value < 0:
    #         value += np.int64(1)
    #     return value

    # value = np.uint64(value)
    # if value >= 2**(size-1):
    #     pad_bits = np.uint8(64 - size)
    #     print('x', value, type(value))
    #     value = np.int64(value << pad_bits) >> pad_bits
    #     print('y', value, type(value))
    #     value += np.int64(1)
    #     #value = (np.uint64(value) << np.uint8(64 - size)).view(np.int64) >> np.uint8(64-size)) + np.int64(1)
    # return value
    # if (value & (1 << (size - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
    #     value = value - (1 << size) + 1  # compute negative value, add one
    # return value                         # return positive value as is

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
