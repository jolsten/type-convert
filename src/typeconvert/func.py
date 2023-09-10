from typeconvert_ext.func import (
    onescomp as _onescomp,
    twoscomp as _twoscomp,
    milstd1750a32 as _milstd1750a32,
    milstd1750a48 as _milstd1750a48,
)
from .utils import validate_unsigned_integer


def onescomp(value: int, size: int) -> int:
    validate_unsigned_integer(value)
    validate_unsigned_integer(size)
    return _onescomp(value, size)


def twoscomp(value: int, size: int) -> int:
    validate_unsigned_integer(value)
    validate_unsigned_integer(size)
    return _twoscomp(value, size)


def milstd1750a32(value: int) -> float:
    validate_unsigned_integer(value, max_value=2**32 - 1)
    return _milstd1750a32(value)


def milstd1750a48(value: int) -> float:
    validate_unsigned_integer(value, max_value=2**48 - 1)
    return _milstd1750a48(value)
