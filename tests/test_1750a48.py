import pytest
import numpy as np
from typeconvert.func import milstd1750a48 as func
from typeconvert.ufunc import milstd1750a48 as ufunc

TEST_ARRAY_SIZE = 100
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

tests = []
for val_in, val_out in TEST_CASES:
    tests.append((val_in, pytest.approx(val_out)))


@pytest.mark.parametrize("val_in, val_out", tests)
def test_func(val_in, val_out):
    print(f"func({val_in:012x}) == {val_out}")
    assert func(val_in) == val_out


@pytest.mark.parametrize("val_in, val_out", tests)
def test_ufunc(val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=">u8")
    expected = [val_out] * TEST_ARRAY_SIZE
    result = ufunc(data)
    print(f"{data[0]:08x} -> {result[0]}")
    assert list(result) == expected
