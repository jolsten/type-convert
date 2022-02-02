import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u8)',
]

from .twoscomp import jfunc as uint_to_twoscomp

def func(value: np.uint64) -> np.float64:
    # Reference:
    # Telemetry Standards, RCC Standard 106-20 Chapter 9, July 2020
    value = np.uint64(value)

    e = uint_to_twoscomp((value >> np.uint8(32)) & np.uint64(0xFF), np.uint8(8))
    s = (value >> np.uint8(31)) & np.uint64(1)
    m = (value & np.uint64(0x007FFFFFFF))

    if e == np.int64(-128):
        return np.float64(0)

    S = np.float64(-2) ** s
    E = np.float64(e)
    M = np.float64(m)

    return ( S + M/np.float64(2**31) ) * np.float64(2) ** E

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
