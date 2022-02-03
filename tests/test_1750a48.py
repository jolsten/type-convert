import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.milstd1750a48 import func, jfunc, ufunc

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
    tests.append((np.uint64(val_in), pytest.approx(val_out)))


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
