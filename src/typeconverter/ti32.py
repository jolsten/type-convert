import numpy as np
from numba import njit, vectorize

from .types import UnsignedInteger, SignedInteger

signatures = [
    'f8(u4)',
]

from typeconverter.twoscomp import jfunc as uint_to_twoscomp

def func(value: np.uint32) -> np.float64:
    value = np.uint32(value)

    e = uint_to_twoscomp((value & np.uint32(0xFF000000)) >> np.uint8(24), np.uint8( 8))
    m = uint_to_twoscomp((value & np.uint32(0x00FFFFFF))                , np.uint8(24))

    E = np.float64(e)
    M = np.float64(m) / np.float64(2**23)

    return M * np.float64(2) ** E

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
