import numpy as np
from numba import njit, vectorize
from .twoscomp import jfunc as uint_to_twoscomp

signatures = [
    'f4(u4)',
]

# Reference(s):
# http://www.xgc-tek.com/manuals/mil-std-1750a/c191.html


def func(value: np.uint32) -> np.float32:
    r"""Convert uint to 1750A32

    Interprets an unsigned integer as a MIL-STD-1750A 32-bit Float and returns
    an IEEE 32-bit Float.

    Parameters
    ----------
    value : unsigned integer
        Unsigned integer value of the data.

    Returns
    -------
    np.float32
        A float containing the interpretation of `value`.

    Examples
    --------
    >>> out = func(0x40000000)
    >>> type(out), out
    (<class 'numpy.float32'>, 0.5)
    """
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
