import numpy as np
from numpy.typing import DTypeLike
from typeconvert._c.ufunc import (
    onescomp as _onescomp,
    twoscomp as _twoscomp,
    milstd1750a32 as _milstd1750a32,
    milstd1750a48 as _milstd1750a48,
    ti32 as _ti32,
    ti40 as _ti40,
    ibm32 as _ibm32,
    ibm64 as _ibm64,
    dec32 as _dec32,
    dec64 as _dec64,
    dec64g as _dec64g,
    bcd as _bcd,
)


def _validate_ndarray(array: np.ndarray, dtype: DTypeLike) -> np.ndarray:
    if not np.issubdtype(array.dtype, dtype):
        array = np.asarray(array, dtype=dtype)
    return array


def onescomp(array: np.ndarray, size: int) -> np.ndarray:
    array = _validate_ndarray(array, np.unsignedinteger)
    return _onescomp(array, np.uint8(size))


def twoscomp(array: np.ndarray, size: int) -> np.ndarray:
    array = _validate_ndarray(array, np.unsignedinteger)
    return _twoscomp(array, np.uint8(size))


def milstd1750a32(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint32)
    return _milstd1750a32(array)


def milstd1750a48(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint64)
    return _milstd1750a48(array)


def ti32(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint32)
    return _ti32(array)


def ti40(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint64)
    return _ti40(array)


def ibm32(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint32)
    return _ibm32(array)


def ibm64(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint64)
    return _ibm64(array)


def dec32(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint32)
    return _dec32(array)


def dec64(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint64)
    return _dec64(array)


def dec64g(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.uint64)
    return _dec64g(array)


def bcd(array: np.ndarray) -> np.ndarray:
    array = _validate_ndarray(array, np.unsignedinteger)
    return _bcd(array)
