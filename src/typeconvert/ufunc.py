import numpy as np
from .utils import validate_unsigned_integer

try:
    from typeconvert_ext.ufunc import onescomp as _onescomp
except ImportError:
    from typeconvert.py.onescomp import ufunc as _onescomp

try:
    from typeconvert_ext.ufunc import twoscomp as _twoscomp
except ImportError:
    from typeconvert.py.twoscomp import ufunc as _twoscomp

try:
    from typeconvert_ext.ufunc import milstd1750a32 as _milstd1750a32
except ImportError:
    from typeconvert.py.milstd1750a32 import ufunc as _milstd1750a32

try:
    from typeconvert_ext.ufunc import milstd1750a48 as _milstd1750a48
except ImportError:
    from typeconvert.py.milstd1750a48 import ufunc as _milstd1750a48


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
