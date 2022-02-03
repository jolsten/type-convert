import pytest
import numpy as np
from itertools import zip_longest
from typeconverter.types.dec32 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100

S = (0b0, 0b1)
E = (0b00000000, 0b11111111)
M = (0x00000000, 0x007FFFFF, 0x003FFFFF)

tests = []
for s in S:
    for e in E:
        for m in M:
            val_in = np.uint32((s << 31) + (e << 23) + m)
            val_out = (-1)**s * (m / 2**24) * 2**(e - 128)
            tests.append((val_in, pytest.approx(val_out)))


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
