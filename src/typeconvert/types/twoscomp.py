import numpy as np
from numba import njit, vectorize
from ..typing import UnsignedInteger, SignedInteger

signatures = [
    'i1(u1,u1)',
    'i2(u2,u1)',
    'i4(u4,u1)',
    'i8(u8,u1)',
]


def func(value: UnsignedInteger, size: np.uint8) -> SignedInteger:
    r"""Convert uint to Two's Complemented int

    Converts an up-to 64-bit uint to a Two's Complement int.

    Parameters
    ----------
    value : unsigned integer
        Unsigned integer value of the data.
    size : np.uint8
        Size of the word in bits.

    Returns
    -------
    np.int64
        A signed integer containing the interpretation of `value`.

    Examples
    --------
    >>> out = func(0xFFF, 12)
    >>> type(out), out
    (<class 'numpy.int64'>, -1)
    """
    value = np.uint64(value)
    if value >= 2**(size-1):
        pad_bits = np.uint8(64 - size)
        value = np.int64(np.uint64(value) << pad_bits) >> pad_bits
    return value


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
