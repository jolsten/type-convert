import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.utils import bits_to_wordsize, mask
from typeconvert.types.twoscomp import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100
TEST_CASES = {
    3: [
        (0b000,  0),
        (0b001,  1),
        (0b010,  2),
        (0b011,  3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ],
    8: [
        (0b00000000,    0),
        (0b00000001,    1),
        (0b00000010,    2),
        (0b01111110,  126),
        (0b01111111,  127),
        (0b10000000, -128),
        (0b10000001, -127),
        (0b10000010, -126),
        (0b11111110,   -2),
        (0b11111111,   -1),
    ],
    16: [
        (0x0000,  0),
        (0xFFFF, -1),
    ],
    24: [
        (0x000000,  0),
        (0xFFFFFF, -1),
    ],
    32: [
        (0x00000000,  0),
        (0xFFFFFFFF, -1),
    ],
    48: [
        (0x000000000000,  0),
        (0xFFFFFFFFFFFF, -1),
    ],
    64: [
        (0x0000000000000000,  0),
        (0xFFFFFFFFFFFFFFFF, -1),
    ],
}

tests = []
for size in TEST_CASES:
    for val_in, val_out in TEST_CASES[size]:
        tests.append((size, val_in, val_out))
tests += [(size, 0, 0) for size in np.arange(64)+1]
tests += [(size, mask(size), -1) for size in np.arange(64)+1]


@pytest.mark.parametrize('size, val_in, val_out', tests)
def test_func(size, val_in, val_out):
    print('func')
    assert func(val_in, size) == val_out


@pytest.mark.parametrize('size, val_in, val_out', tests)
def test_njit(size, val_in, val_out):
    print('jfunc')
    size = np.uint8(size)
    iter = {8: np.uint8, 16: np.uint16, 32: np.uint32, 64: np.uint64}
    for bits, dtype in iter.items():
        if size <= bits:
            break
    val_in = dtype(val_in)
    print('C', val_in, val_in.dtype, size, size.dtype)
    assert jfunc(val_in, size) == val_out


@pytest.mark.parametrize('size, val_in, val_out', tests)
def test_vectorize(size, val_in, val_out):
    print('ufunc')
    data = np.array(
        [val_in] * TEST_ARRAY_SIZE, dtype=f'uint{bits_to_wordsize(size)}'
    )
    expected = [val_out] * TEST_ARRAY_SIZE
    assert all([a == b for a, b in zip_longest(ufunc(data, size), expected)])
