import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.ibm64 import func, jfunc, ufunc

# Reference:
# https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point

TEST_ARRAY_SIZE = 100
TEST_CASES = [
    (0x0010000000000000,            5.397605e-79),
    (0x8010000000000000,           -5.397605e-79),
    (0x0000000000000000,                     0.0),
    (0x401999999999999A,                     0.1),
    (0xC01999999999999A,                    -0.1),
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
