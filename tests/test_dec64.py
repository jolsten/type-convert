import pytest
import numpy as np
from numba import njit, vectorize
from itertools import zip_longest

from typeconverter.utils import bits_to_wordsize, mask
from typeconverter.dec64 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100

S = (0b0, 0b1)
E = (0b00000000, 0b11111111)
M = (0x0000000000000000, 0x003FFFFFFFFFFFFF, 0x007FFFFFFFFFFFFF)

tests = []
for s in S:
    for e in E:
        for m in M:
            val_in = np.uint64( (s<<63) + (e<<55) + m )
            val_out = (-1)**s * (m / 2**56) * 2**(e - 128)
            tests.append( (val_in, pytest.approx(val_out)) )


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
    ufunc(data)
    assert all([a==b for a, b in zip_longest(ufunc(data), [val_out] * TEST_ARRAY_SIZE)])
