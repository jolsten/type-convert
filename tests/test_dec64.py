import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.dec64 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100

TEST_CASES = [
    (0x4080000000000000,       1.000000000000000),
    (0xC080000000000000,      -1.000000000000000),
    (0x4160000000000000,       3.500000000000000),
    (0xC160000000000000,      -3.500000000000000),
    (0x41490FDAA22168BE,       3.141592653589793),
    (0xC1490FDAA22168BE,      -3.141592653589793),
    (0x7DF0BDC21ABB48DB,  1.0000000000000000E+37),
    (0xFDF0BDC21ABB48DB, -1.0000000000000000E+37),
    (0x03081CEA14545C75,  9.9999999999999999E-38),
    (0x83081CEA14545C75, -9.9999999999999999E-38),
    (0x409E06521462CEE7,       1.234567890123450),
    (0xC09E06521462CEE7,      -1.234567890123450),
]

tests = []
for val_in, val_out in TEST_CASES:
    tests.append((np.uint64(val_in), pytest.approx(val_out)))


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
