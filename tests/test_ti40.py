import pytest
import numpy as np
from numba import njit, vectorize
from itertools import zip_longest

from typeconverter.utils import bits_to_wordsize, mask
from typeconverter.ti40 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100

E = [0x7F, 0x00, 0xFF]
M = [0x007FFFFFFF, 0x0000000000, 0x00FFFFFFFF]

from typeconverter.twoscomp import func as uint_to_twoscomp
tests = []
for e in E:
    for m in M:
        val_in = np.uint64( (e << 32) + m )
        val_out = (uint_to_twoscomp(m, 32) / 2**31) * np.float64(2) ** uint_to_twoscomp(e, 8)
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
