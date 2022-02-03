import numpy as np
from numba import njit, vectorize
from .twoscomp import jfunc as uint_to_twoscomp

signatures = [
    'f8(u4)',
]


def func(value: np.uint32) -> np.float64:
    # Reference:
    # https://www.ti.com/lit/an/spra400/spra400.pdf
    value = np.uint32(value)

    e = uint_to_twoscomp(
        (value & np.uint32(0xFF000000)) >> np.uint8(24), np.uint8(8)
    )
    s = (value & np.uint32(0x00800000)) >> np.uint8(23)
    m = (value & np.uint32(0x007FFFFF))

    if e == np.int64(-128):
        return np.float64(0)

    S = np.float64(-2) ** s
    E = np.float64(e)
    M = np.float64(m)

    return (S + M/np.float64(2**23)) * np.float64(2) ** E


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
