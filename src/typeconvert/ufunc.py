import numpy as np
from typeconvert_ext.ufunc import onescomp as _onescomp, twoscomp as _twoscomp
from .utils import validate_unsigned_integer


def onescomp(data: np.ndarray, size: int) -> np.ndarray:
    validate_unsigned_integer(size)
    return _onescomp(data, np.uint8(size))


def twoscomp(data: np.ndarray, size: int) -> np.ndarray:
    validate_unsigned_integer(size)
    return _twoscomp(data, np.uint8(size))
