import pytest
import numpy as np
from itertools import zip_longest
from typeconvert.types.ti32 import func, jfunc, ufunc

TEST_ARRAY_SIZE = 100
# References:
# https://www.ti.com/lit/an/spra400/spra400.pdf
# https://stackoverflow.com/questions/64687130/convert-ti-tms320c30-32-bits-float-to-ieee-float-in-python
TEST_CASES = (
    (0x7F, 0, 0b11111111111111111111111, (2-2**-23) * 2**127),
    (0x7F, 0, 0b11111111111111111111110, (2-2**-22) * 2**127),
    (0x7F, 0, 0b11111111111111111111101, (2-2**-21+2**-23) * 2**127),
    (0x7F, 0, 0b11111111111111111111100, (2-2**-21) * 2**127),

    (0x7F, 0, 0b00000000000000000000000, 2**127),
    (0x7E, 0, 0b11111111111111111111111, (2-2**-23) * 2**126),
    (0x7E, 0, 0b11111111111111111111110, (2-2**-22) * 2**126),
    (0x7E, 0, 0b11111111111111111111101, (2-2**-21+2**-23) * 2**126),

    (0x00, 0, 0b00000000000000000000000, 1),
    (0xFF, 0, 0b11111111111111111111111, 1-2**-24),
    (0xFF, 0, 0b11111111111111111111110, 1-2**-23),
    (0xFF, 0, 0b11111111111111111111101, 1-2**-22+2**-24),

    (0xFF, 0, 0b00000000000000000000000, 2**-1),
    (0xFE, 0, 0b11111111111111111111111, (2-2**-23) * 2**-2),
    (0xFE, 0, 0b11111111111111111111110, (2-2**-22) * 2**-2),
    (0xFE, 0, 0b11111111111111111111101, (2-2**-21+2**-23) * 2**-2),

    (0x82, 0, 0b00000000000000000000000, 2**-126),
    (0x81, 0, 0b11111111111111111111111, (2-2**-23) * 2**-127),
    (0x81, 0, 0b11111111111111111111110, (2-2**-22) * 2**-127),
    (0x81, 0, 0b11111111111111111111101, (2-2**-21+2**-23) * 2**-127),
    (0x81, 0, 0b11111111111111111111100, (2-2**-21) * 2**-127),

    (0x81, 0, 0b00000000000000000000010, (1+2**-22) * 2**-127),
    (0x81, 0, 0b00000000000000000000001, (1+2**-23) * 2**-127),
    (0x81, 0, 0b00000000000000000000000, 2**-127),

    # e = -128 implies zero
    (0x80, 0, 0b11111111111111111111111, 0.0),
    (0x80, 0, 0b11111111111111111111110, 0.0),
    (0x80, 0, 0b11111111111111111111101, 0.0),

    (0x80, 0, 0b00000000000000000000001, 0.0),
    (0x80, 0, 0b00000000000000000000000, 0.0),

    (0x80, 1, 0b11111111111111111111111, 0.0),
    (0x80, 1, 0b11111111111111111111110, 0.0),
    (0x80, 1, 0b11111111111111111111101, 0.0),

    (0x80, 1, 0b00000000000000000000011, 0.0),
    (0x80, 1, 0b00000000000000000000010, 0.0),
    (0x80, 1, 0b00000000000000000000001, 0.0),

    (0x80, 1, 0b00000000000000000000000, 0.0),

    (0x81, 1, 0b11111111111111111111111, (-1-2**-23) * 2**-127),
    (0x81, 1, 0b11111111111111111111110, (-1-2**-22) * 2**-127),
    (0x81, 1, 0b11111111111111111111101, (-1-2**-21+2**-23) * 2**-127),

    (0x81, 1, 0b00000000000000000000010, (-2+2**-22) * 2**-127),
    (0x81, 1, 0b00000000000000000000001, (-2+2**-23) * 2**-127),

    (0x81, 1, 0b00000000000000000000000, - 2**-126),
    (0x82, 1, 0b11111111111111111111111, (-1-2**-23) * 2**-126),
    (0x82, 1, 0b11111111111111111111110, (-1-2**-22) * 2**-126),
    (0x82, 1, 0b11111111111111111111101, (-1-2**-21+2**-23) * 2**-126),

    (0xFF, 1, 0b00000000000000000000001, (-1+2**-24)),
    (0xFF, 1, 0b00000000000000000000000, -1.0),

    # These three tests appear to be wrong,
    # exp = 0 should yield 2**0, not 2**-1
    # (0x00, 1, 0b11111111111111111111111, (-1-2**-23) * 2**-1),
    # (0x00, 1, 0b11111111111111111111110, (-1-2**-22) * 2**-1),
    # (0x00, 1, 0b11111111111111111111101, (-1-2**-21+2**-23) * 2**-1), #
    (0xFF, 1, 0b11111111111111111111111, (-1-2**-23) * 2**-1),
    (0xFF, 1, 0b11111111111111111111110, (-1-2**-22) * 2**-1),
    (0xFF, 1, 0b11111111111111111111101, (-1-2**-21+2**-23) * 2**-1),
    (0x00, 1, 0b11111111111111111111111, (-1-2**-23) * 2**0),
    (0x00, 1, 0b11111111111111111111110, (-1-2**-22) * 2**0),
    (0x00, 1, 0b11111111111111111111101, (-1-2**-21+2**-23) * 2**0),

    (0x00, 1, 0b00000000000000000000001, (-2+2**-23)),
    (0x00, 1, 0b00000000000000000000000, -2),
    (0x01, 1, 0b11111111111111111111111, -2-2**-22),
    (0x01, 1, 0b11111111111111111111110, -2-2**-21),
    (0x01, 1, 0b11111111111111111111101, -2-2**-20+2**-22),

    (0x7F, 1, 0b00000000000000000000001, (-2+2**-23) * 2**127),
    (0x7F, 1, 0b00000000000000000000000, - 2**128),
)

tests = []
for e, s, m, val_out in TEST_CASES:
    val_in = np.uint32((e << 24) + (s << 23) + m)
    tests.append((val_in, pytest.approx(val_out)))


@pytest.mark.parametrize('val_in, val_out', tests)
def test_func(val_in, val_out):
    print('func')
    print(f'val_in = {val_in:08x}')
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
