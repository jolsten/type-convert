import numpy as np

# from numba import njit, vectorize

signatures = [
    "u8(i8)",
]


def func(value: np.uint64) -> np.int64:
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
    result = np.int64(0)
    place = np.uint64(1)
    digit = np.uint8(0)
    while value > np.uint64(0):
        digit = np.uint8(value) & np.uint8(0xF)
        if digit >= 10:
            result = np.int64(-1)
            value = np.uint64(0)
        else:
            result += digit * place
            place *= np.uint64(10)
            value = np.uint64(value) >> np.uint8(4)

    return result


# jfunc = njit(signatures)(func)
# ufunc = vectorize(signatures)(func)
