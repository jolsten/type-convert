import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.dec32 import func, jfunc, ufunc

# https://pubs.usgs.gov/of/2005/1424/

TEST_ARRAY_SIZE = 100
TEST_CASES = [
    # F4
    (0x40800000,       1.000000),
    (0xC0800000,      -1.000000),
    (0x41600000,       3.500000),
    (0xC1600000,      -3.500000),
    (0x41490FD0,       3.141590),
    (0xC1490FD0,      -3.141590),
    (0x7DF0BDC2,  9.9999999E+36),
    (0xFDF0BDC2, -9.9999999E+36),
    (0x03081CEA,  9.9999999E-38),
    (0x83081CEA, -9.9999999E-38),
    (0x409E0652,       1.234568),
    (0xC09E0652,      -1.234568),
    (0x7FFFFFFF,  1.7014118e+38),  # last two not from reference
    (0xFFFFFFFF, -1.7014118e+38),
]

tests = []
for val_in, val_out in TEST_CASES:
    tests.append((np.uint32(val_in), pytest.approx(val_out)))


@pytest.mark.parametrize('val_in, val_out', tests)
def test_func(val_in, val_out):
    print('func', val_in, val_out)
    assert func(val_in) == val_out


@pytest.mark.parametrize('val_in, val_out', tests)
def test_njit(val_in, val_out):
    print('jfunc', val_in, val_out)
    assert jfunc(val_in) == val_out


@pytest.mark.parametrize('val_in, val_out', tests)
def test_vectorize(val_in, val_out):
    print('ufunc', val_in, val_out)
    data = np.array([val_in] * TEST_ARRAY_SIZE)
    expected = [val_out] * TEST_ARRAY_SIZE
    assert all([a == b for a, b in zip_longest(ufunc(data), expected)])
