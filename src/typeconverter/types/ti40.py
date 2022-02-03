import numpy as np
from numba import njit, vectorize
from .twoscomp import jfunc as uint_to_twoscomp

signatures = [
    'f8(u8)',
]


def func(value: np.uint64) -> np.float64:
    r"""Convert uint to TI40

    Interprets an unsigned integer as a Texas Instruments 40-bit Float and
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
    >>> out = func(0xFF80000000)
    >>> type(out), out
    (<class 'numpy.float64'>, -1.0)
    """
    # Reference:
    # Telemetry Standards, RCC Standard 106-20 Chapter 9, July 2020
    value = np.uint64(value)

    e = uint_to_twoscomp(
        (value >> np.uint8(32)) & np.uint64(0xFF), np.uint8(8)
    )
    s = (value >> np.uint8(31)) & np.uint64(1)
    m = (value & np.uint64(0x007FFFFFFF))

    if e == np.int64(-128):
        return np.float64(0)

    S = np.float64(-2) ** s
    E = np.float64(e)
    M = np.float64(m)

    return (S + M/np.float64(2**31)) * np.float64(2) ** E


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
