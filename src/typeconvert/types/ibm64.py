import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u8)',
]


def func(value: np.uint64) -> np.float64:
    r"""Convert uint to IBM64

    Interprets an unsigned integer as a IBM 64-bit Float and
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
    >>> out = func(0x4019999A)
    >>> type(out), out
    (<class 'numpy.float64'>, 0.1)
    """
    value = np.uint64(value)

    s = (value >> np.uint8(63)) * np.uint64(1)
    e = (value >> np.uint8(56)) & np.uint64(0x7F)
    m = (value & np.uint64(0x00FFFFFFFFFFFFFF))

    S = np.int8(-1) ** s
    E = np.int8(e) - np.int8(64)
    M = np.float64(m) / np.float64(2**56)

    return np.float64(S * M * np.float64(16)**E)


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
