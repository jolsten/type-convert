from typing import Optional
import numpy as np


def bits_to_wordsize(size: np.uint8) -> np.uint8:
    for word_size in [8, 16, 32, 64]:
        if size <= word_size:
            return word_size
    return 64


def mask(size: int) -> int:
    mask = 1
    for _ in range(size - 1):
        mask = (mask << 1) + 1
    return mask


def validate_unsigned_integer(
    value: int, min_value: int = 0, max_value: Optional[int] = None
) -> None:
    if not isinstance(value, int):
        raise TypeError(f"argument must be an integer")

    if min_value is not None and value < min_value:
        raise ValueError(f"argument must be >= max value {min_value}")

    if max_value is not None and value > max_value:
        raise ValueError(f"argument must be <= max value {max_value}")
