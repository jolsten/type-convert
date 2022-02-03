import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.milstd1750a32 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100
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
    tests.append((np.uint32(val_in), pytest.approx(val_out)))


@pytest.mark.parametrize('val_in, val_out', tests)
def test_func(val_in, val_out):
    print('func')
    assert func(val_in) == val_out


@pytest.mark.parametrize('val_in, val_out', tests)
def test_njit(val_in, val_out):
    print('jfunc', val_in, val_out)
    assert jfunc(val_in) == val_out


@pytest.mark.parametrize('val_in, val_out', tests)
def test_vectorize(val_in, val_out):
    print('ufunc')
    data = np.array([val_in] * TEST_ARRAY_SIZE)
    expected = [val_out] * TEST_ARRAY_SIZE
    assert all([a == b for a, b in zip_longest(ufunc(data), expected)])
