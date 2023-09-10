import pytest
import numpy as np
from typeconvert.func import milstd1750a32 as func
from typeconvert.ufunc import milstd1750a32 as ufunc

TEST_ARRAY_SIZE = 1
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

tests = []
for val_in, val_out in TEST_CASES:
    tests.append((val_in, pytest.approx(val_out)))


@pytest.mark.parametrize("val_in, val_out", tests)
def test_func(val_in, val_out):
    assert func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", tests)
def test_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u4")
    expected = [val_out] * TEST_ARRAY_SIZE
    result = ufunc(data)
    print(f"{data[0]:08x} -> {result[0]}")
    assert list(result) == expected
