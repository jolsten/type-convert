import numpy as np

from .types import UnsignedInteger, SignedInteger

def uint_to_twoscomp(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
    if (value & (1 << (size - 1))) != 0: # if sign bit is set e.g., 8bit: 128-255
        value = value - (1 << size)      # compute negative value
    return value                         # return positive value as is
