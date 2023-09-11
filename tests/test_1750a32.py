import pytest
import numpy as np
from typeconvert.py.func import milstd1750a32 as py_func
from typeconvert.py.ufunc import milstd1750a32 as py_ufunc
from typeconvert.c.func import milstd1750a32 as c_func
from typeconvert.c.ufunc import milstd1750a32 as c_ufunc
from .conftest import TEST_ARRAY_SIZE


TEST_CASES = [
    (0x7FFFFF7F, 0.9999998 * 2**127),
    (0x4000007F, 0.5 * 2**127),
    (0x50000004, 0.625 * 2**4),
    (0x40000001, 0.5 * 2**1),
    (0x40000000, 0.5 * 2**0),
    (0x400000FF, 0.5 * 2**-1),
    (0x40000080, 0.5 * 2**-128),
    (0x00000000, 0.0 * 2**0),
    (0x80000000, -1.0 * 2**0),
    (0xBFFFFF80, -0.5000001 * 2**-128),
    (0x9FFFFF04, -0.7500001 * 2**4),
]
TEST_CASES = [(a, pytest.approx(b)) for a, b in TEST_CASES]


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_py_func(val_in, val_out):
    assert py_func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_py_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u4")
    assert list(py_ufunc(data)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_c_func(val_in, val_out):
    assert c_func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_c_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u4")
    assert list(c_ufunc(data)) == [val_out] * TEST_ARRAY_SIZE
