import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u8)',
]


def func(value: np.uint64) -> np.float64:
    r"""Convert uint to DEC64

    Interprets an unsigned integer as a DEC 64-bit Float and
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
    >>> out = func(0x409E06521462CEE7)
    >>> type(out), out
    (<class 'numpy.float64'>, 1.234567890123450)
    """
    value = np.uint64(value)

    s = (value >> np.uint8(63)) & np.uint64(1)
    e = (value >> np.uint8(55)) & np.uint64(0xFF)
    m = (value & np.uint64(0x007FFFFFFFFFFFFF))

    S = np.int8(-1) ** s
    E = np.int16(e) - np.int16(128)
    M = np.float64(m) / np.float64(2**56) + np.float64(0.5)

    return np.float64(S * M * np.float64(2)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
