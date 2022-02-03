import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.bcd import func, jfunc, ufunc

# https://pubs.usgs.gov/of/2005/1424/
TEST_ARRAY_SIZE = 100
digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
np.random.seed(0)
TEST_CASES = np.random.choice(digits, size=(100, 16))

tests = []
for row in TEST_CASES:
    hex_str = ''.join([str(x) for x in row])
    val_in = np.uint64(int(hex_str, base=16))
    val_out = np.uint64(int(hex(val_in)[2:], base=10))
    tests.append((val_in, val_out))


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
