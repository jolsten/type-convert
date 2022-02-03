import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u4)',
]


def func(value: np.uint32) -> np.float64:
    r"""Convert uint to DEC32

    Interprets an unsigned integer as a DEC 32-bit Float and
    returns an IEEE 64-bit Float.

    Parameters
    ----------
    value : unsigned integer
        Unsigned integer value of the data.

    Returns
    -------
    np.float64
        A float containing the interpretation of `value`.

    Examples
    --------
    >>> out = func(0x409E0652)
    >>> type(out), out
    (<class 'numpy.float64'>, 1.234568)
    """
    value = np.uint32(value)

    s = (value >> np.uint8(31)) * np.uint32(1)
    e = (value >> np.uint8(23)) & np.uint32(0xFF)
    m = (value & np.uint32(0x007FFFFF))

    S = np.int8(-1) ** s
    E = np.int16(e) - np.int16(128)
    M = np.float64(m) / np.float64(2**24) + np.float64(0.5)

    return np.float64(S * M * np.float64(2)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
