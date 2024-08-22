from typing import Optional

import numpy as np
import packaging.version as pp
from numpy.typing import DTypeLike

NPY_CAST_SAFE = pp.parse(np.__version__) >= pp.parse("1.23")


def bits_to_wordsize(size: np.uint8) -> np.uint8:
    for word_size in [8, 16, 32, 64]:
        if size <= word_size:
            return word_size
    return 64


def _bits_to_dtype(size: int) -> DTypeLike:
    return f"uint{bits_to_wordsize(size)}"


def mask(size: int) -> int:
    mask = 1
    for _ in range(size - 1):
        mask = (mask << 1) + 1
    return mask


def _validate_unsigned_integer(
    value: int,
    min_value: int = 0,
    max_value: Optional[int] = None,
    bits: Optional[int] = None,
) -> None:
    if not isinstance(value, int):
        msg = "argument must be an integer"
        raise TypeError(msg)

    if min_value is not None and value < min_value:
        msg = f"argument must be >= max value {min_value}"
        raise ValueError(msg)

    if max_value and bits:
        msg = "Must provide either max_value or bits, not both"
        raise ValueError(msg)

    if bits is not None:
        max_value = 2**bits - 1

    if max_value is not None and value > max_value:
        msg = f"argument must be <= max value {max_value}"
        raise ValueError(msg)
