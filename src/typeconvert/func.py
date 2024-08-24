from typeconvert._c.func import bcd as _bcd
from typeconvert._c.func import dec32 as _dec32
from typeconvert._c.func import dec64 as _dec64
from typeconvert._c.func import dec64g as _dec64g
from typeconvert._c.func import ibm32 as _ibm32
from typeconvert._c.func import ibm64 as _ibm64
from typeconvert._c.func import milstd1750a32 as _milstd1750a32
from typeconvert._c.func import milstd1750a48 as _milstd1750a48
from typeconvert._c.func import onescomp as _onescomp
from typeconvert._c.func import ti32 as _ti32
from typeconvert._c.func import ti40 as _ti40
from typeconvert._c.func import twoscomp as _twoscomp
from typeconvert.utils import _validate_unsigned_integer


def onescomp(value: int, size: int) -> int:
    _validate_unsigned_integer(value)
    _validate_unsigned_integer(size)
    return _onescomp(value, size)


def twoscomp(value: int, size: int) -> int:
    _validate_unsigned_integer(value)
    _validate_unsigned_integer(size)
    return _twoscomp(value, size)


def milstd1750a32(value: int) -> float:
    _validate_unsigned_integer(value, bits=32)
    return _milstd1750a32(value)


def milstd1750a48(value: int) -> float:
    _validate_unsigned_integer(value, bits=48)
    return _milstd1750a48(value)


def ti32(value: int) -> float:
    _validate_unsigned_integer(value, bits=32)
    return _ti32(value)


def ti40(value: int) -> float:
    _validate_unsigned_integer(value, bits=40)
    return _ti40(value)


def ibm32(value: int) -> float:
    _validate_unsigned_integer(value, bits=32)
    return _ibm32(value)


def ibm64(value: int) -> float:
    _validate_unsigned_integer(value, bits=64)
    return _ibm64(value)


def dec32(value: int) -> float:
    _validate_unsigned_integer(value, bits=32)
    return _dec32(value)


def dec64(value: int) -> float:
    _validate_unsigned_integer(value, bits=64)
    return _dec64(value)


def dec64g(value: int) -> float:
    _validate_unsigned_integer(value, bits=64)
    return _dec64g(value)


def bcd(value: int) -> int:
    _validate_unsigned_integer(value)
    return _bcd(value)
