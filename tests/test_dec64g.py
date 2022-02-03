import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.dec64g import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100

TEST_CASES = [
    (0x4010000000000000,       1.000000000000000),
    (0xC010000000000000,      -1.000000000000000),
    (0x402C000000000000,       3.500000000000000),
    (0xC02C000000000000,      -3.500000000000000),
    (0x402921FB54442D18,       3.141592653589793),
    (0xC02921FB54442D18,      -3.141592653589793),
    (0x47BE17B84357691B,  1.0000000000000000E+37),
    (0xC7BE17B84357691B, -1.0000000000000000E+37),
    (0x3861039D428A8B8F,  9.9999999999999999E-38),
    (0xB861039D428A8B8F, -9.9999999999999999E-38),
    (0x4013C0CA428C59DD,       1.234567890123450),
    (0xC013C0CA428C59DD,      -1.234567890123450),
]

tests = []
for val_in, val_out in TEST_CASES:
    tests.append((np.uint64(val_in), pytest.approx(val_out)))


@pytest.mark.parametrize('val_in, val_out', tests)
def test_func(val_in, val_out):
    print('func')
    print(f'val_in = {val_in:016x}')
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
