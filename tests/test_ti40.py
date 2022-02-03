import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.ti40 import func, jfunc, ufunc
from typeconvert.types.twoscomp import func as uint_to_twoscomp

TEST_ARRAY_SIZE = 100

S = [0, 1]
E = [0x7F, 0x7E, 0x01, 0xFF]
M = [0x007FFFFFFF, 0x007FFFFFFE, 0x007FFFFFFD,
     0x0000000002, 0x0000000001, 0x0000000000]

tests = []
for s in S:
    for e in E:
        for m in M:
            val_in = np.uint64((e << 32) | (s << 31) | m)
            val_out = ((-2)**s + float(m) / 2**31) \
                * float(2)**uint_to_twoscomp(e, 8)
            tests.append((val_in, pytest.approx(val_out)))


@pytest.mark.parametrize('val_in, val_out', tests)
def test_func(val_in, val_out):
    print('func')
    print(f'val_in = {val_in:010x}')
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
