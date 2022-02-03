import numpy as np
from numba import njit, vectorize

signatures = [
    'u8(u8)',
]


def func(value: np.uint64) -> np.uint64:
    r"""Convert Binary-Coded Decimal (BCD) to an uint.

    Converts an up-to 64-bit BCD value to an unsigned integer.

    Parameters
    ----------
    value : np.uint64
        Unsigned integer value of the BCD-encoded data.

    Returns
    -------
    np.uint64
        An unsigned integer of the decoded BCD data.

    Examples
    --------
    >>> import numpy as np
    >>> bcd_value = np.uint64(0x12345678)
    >>> out = func(bcd_value)
    >>> type(out), out
    (<class 'numpy.uint64'>, 12345678)
    """
    value = np.uint64(value)

    out = np.uint64(0)
    idx = np.uint64(0)
    while value:
        out += (
                (value & np.uint8(0xF)) % np.uint8(10)
            ) * np.uint64(10) ** np.uint8(idx)
        value = value >> np.uint8(4)
        idx += 1
    return out


jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
