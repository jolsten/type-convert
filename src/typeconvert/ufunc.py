import numpy as np
from .utils import validate_unsigned_integer
from typeconvert._c.ufunc import onescomp as _onescomp
from typeconvert._c.ufunc import twoscomp as _twoscomp
from typeconvert._c.ufunc import milstd1750a32 as _milstd1750a32
from typeconvert._c.ufunc import milstd1750a48 as _milstd1750a48


def onescomp(data: np.ndarray, size: int) -> np.ndarray:
    validate_unsigned_integer(size)
    return _onescomp(data, np.uint8(size))


def twoscomp(data: np.ndarray, size: int) -> np.ndarray:
    validate_unsigned_integer(size)
    return _twoscomp(data, np.uint8(size))


def milstd1750a32(data: np.ndarray) -> np.ndarray:
    if data.dtype != np.uint32:
        data = np.asarray(data, dtype="uint32")
    return _milstd1750a32(data)


def milstd1750a48(data: np.ndarray) -> np.ndarray:
    if data.dtype != np.uint64:
        data = np.asarray(data, dtype="uint64")
    return _milstd1750a48(data)
