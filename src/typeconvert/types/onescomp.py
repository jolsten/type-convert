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
    r"""Convert uint to One's Complemented int

    Converts an up-to 64-bit uint to a One's Complement int.

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
    >>> out = func(0xFFE, 12)
    >>> type(out), out
    (<class 'numpy.int64'>, -1)
    """
    value = np.uint64(value)
    pad_bits = np.uint8(64-size)
    if value & (np.uint64(1) << np.uint8(size-1)) != 0:
        return -np.int64((~(value << pad_bits)) >> pad_bits)
    return value


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
