import pytest
import numpy as np
from typeconvert.py.func import milstd1750a48 as py_func
from typeconvert.py.ufunc import milstd1750a48 as py_ufunc
from typeconvert.c.func import milstd1750a48 as c_func
from typeconvert.c.ufunc import milstd1750a48 as c_ufunc
from .conftest import TEST_ARRAY_SIZE

TEST_CASES = [
    (0x4000007F0000, 0.5 * 2**127),
    (0x400000000000, 0.5 * 2**0),
    (0x400000FF0000, 0.5 * 2**-1),
    (0x400000800000, 0.5 * 2**-128),
    (0x8000007F0000, -1.0 * 2**127),
    (0x800000000000, -1.0 * 2**0),
    (0x800000FF0000, -1.0 * 2**-1),
    (0x800000800000, -1.0 * 2**-128),
    (0x000000000000, 0.0 * 2**0),
    (0xA00000FF0000, -0.75 * 2**-1),
]
TEST_CASES = [(a, pytest.approx(b)) for a, b in TEST_CASES]


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_py_func(val_in, val_out):
    assert py_func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_py_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u8")
    assert list(py_ufunc(data)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_c_func(val_in, val_out):
    assert c_func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
def test_c_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u8")
    assert list(c_ufunc(data)) == [val_out] * TEST_ARRAY_SIZE
