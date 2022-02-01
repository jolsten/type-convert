import numpy as np
from numba import njit, vectorize

from .types import UnsignedInteger, SignedInteger

signatures = [
    'f8(u8)',
]

from typeconverter.twoscomp import jfunc as uint_to_twoscomp

def func(value: np.uint64) -> np.float64:
    value = np.uint64(value)
    
    e = uint_to_twoscomp((value & np.uint64(0xFF00000000)) >> np.uint8(32), np.uint8( 8))
    m = uint_to_twoscomp((value & np.uint64(0x00FFFFFFFF))                , np.uint8(32))

    E = np.float64(e)
    M = np.float64(m) / np.float64(2**31)

    return M * np.float64(2) ** E

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
