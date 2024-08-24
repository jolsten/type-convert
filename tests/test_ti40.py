import numpy as np
import pytest
from typeconvert._c.func import ti40 as c_func
from typeconvert._c.ufunc import ti40 as c_ufunc
from typeconvert._py.func import ti40 as py_func

# from typeconvert._py.ufunc import ti40 as py_ufunc
from typeconvert.func import ti40 as func
from typeconvert.ufunc import ti40 as ufunc

from .conftest import SpecificCasesBase

TEST_FROM_COMPONENTS = (
    (0x7F, 0, 0b1111111111111111111111111111111, (2 - 2**-31) * 2**127),
    (0x7F, 0, 0b1111111111111111111111111111110, (2 - 2**-30) * 2**127),
    (0x7F, 0, 0b1111111111111111111111111111101, (2 - 2**-29 + 2**-31) * 2**127),
    (0x7F, 0, 0b1111111111111111111111111111100, (2 - 2**-29) * 2**127),
    (0x7F, 0, 0b0000000000000000000000000000000, 2**127),
    (0x7E, 0, 0b1111111111111111111111111111111, (2 - 2**-31) * 2**126),
    (0x7E, 0, 0b1111111111111111111111111111110, (2 - 2**-30) * 2**126),
    (0x7E, 0, 0b1111111111111111111111111111101, (2 - 2**-20 + 2**-31) * 2**126),
    (0x00, 0, 0b0000000000000000000000000000000, 1),
    (0xFF, 0, 0b1111111111111111111111111111111, 1 - 2**-24),
    (0xFF, 0, 0b1111111111111111111111111111110, 1 - 2**-31),
    (0xFF, 0, 0b1111111111111111111111111111101, 1 - 2**-30 + 2**-24),
    (0xFF, 0, 0b0000000000000000000000000000000, 2**-1),
    (0xFE, 0, 0b1111111111111111111111111111111, (2 - 2**-31) * 2**-2),
    (0xFE, 0, 0b1111111111111111111111111111110, (2 - 2**-30) * 2**-2),
    (0xFE, 0, 0b1111111111111111111111111111101, (2 - 2**-29 + 2**-31) * 2**-2),
    (0x82, 0, 0b0000000000000000000000000000000, 2**-126),
    (0x81, 0, 0b1111111111111111111111111111111, (2 - 2**-31) * 2**-127),
    (0x81, 0, 0b1111111111111111111111111111110, (2 - 2**-30) * 2**-127),
    (0x81, 0, 0b1111111111111111111111111111101, (2 - 2**-29 + 2**-31) * 2**-127),
    (0x81, 0, 0b1111111111111111111111111111100, (2 - 2**-29) * 2**-127),
    (0x81, 0, 0b0000000000000000000000000000010, (1 + 2**-30) * 2**-127),
    (0x81, 0, 0b0000000000000000000000000000001, (1 + 2**-31) * 2**-127),
    (0x81, 0, 0b0000000000000000000000000000000, 2**-127),
    # e = -128 implies zero
    (0x80, 0, 0b1111111111111111111111111111111, 0.0),
    (0x80, 0, 0b1111111111111111111111111111110, 0.0),
    (0x80, 0, 0b1111111111111111111111111111101, 0.0),
    (0x80, 0, 0b0000000000000000000000000000001, 0.0),
    (0x80, 0, 0b0000000000000000000000000000000, 0.0),
    (0x80, 1, 0b1111111111111111111111111111111, 0.0),
    (0x80, 1, 0b1111111111111111111111111111110, 0.0),
    (0x80, 1, 0b1111111111111111111111111111101, 0.0),
    (0x80, 1, 0b0000000000000000000000000000011, 0.0),
    (0x80, 1, 0b0000000000000000000000000000010, 0.0),
    (0x80, 1, 0b0000000000000000000000000000001, 0.0),
    (0x80, 1, 0b0000000000000000000000000000000, 0.0),
    (0x81, 1, 0b1111111111111111111111111111111, (-1 - 2**-31) * 2**-127),
    (0x81, 1, 0b1111111111111111111111111111110, (-1 - 2**-30) * 2**-127),
    (
        0x81,
        1,
        0b1111111111111111111111111111101,
        (-1 - 2**-29 + 2**-31) * 2**-127,
    ),
    (0x81, 1, 0b0000000000000000000000000000010, (-2 + 2**-30) * 2**-127),
    (0x81, 1, 0b0000000000000000000000000000001, (-2 + 2**-31) * 2**-127),
    (0x81, 1, 0b0000000000000000000000000000000, -(2**-126)),
    (0x82, 1, 0b1111111111111111111111111111111, (-1 - 2**-31) * 2**-126),
    (0x82, 1, 0b1111111111111111111111111111110, (-1 - 2**-30) * 2**-126),
    (
        0x82,
        1,
        0b1111111111111111111111111111101,
        (-1 - 2**-29 + 2**-31) * 2**-126,
    ),
    (0xFF, 1, 0b0000000000000000000000000000001, (-1 + 2**-24)),
    (0xFF, 1, 0b0000000000000000000000000000000, -1.0),
    # These three tests appear to be wrong,
    # exp = 0 should yield 2**0, not 2**-1
    # (0x00, 1, 0b11111111111111111111111, (-1-2**-31) * 2**-1),
    # (0x00, 1, 0b11111111111111111111110, (-1-2**-30) * 2**-1),
    # (0x00, 1, 0b11111111111111111111101, (-1-2**-29+2**-31) * 2**-1), #
    (0xFF, 1, 0b1111111111111111111111111111111, (-1 - 2**-31) * 2**-1),
    (0xFF, 1, 0b1111111111111111111111111111110, (-1 - 2**-30) * 2**-1),
    (0xFF, 1, 0b1111111111111111111111111111101, (-1 - 2**-29 + 2**-31) * 2**-1),
    (0x00, 1, 0b1111111111111111111111111111111, (-1 - 2**-31) * 2**0),
    (0x00, 1, 0b1111111111111111111111111111110, (-1 - 2**-30) * 2**0),
    (0x00, 1, 0b1111111111111111111111111111101, (-1 - 2**-29 + 2**-31) * 2**0),
    (0x00, 1, 0b0000000000000000000000000000001, (-2 + 2**-31)),
    (0x00, 1, 0b0000000000000000000000000000000, -2),
    (0x01, 1, 0b1111111111111111111111111111111, -2 - 2**-30),
    (0x01, 1, 0b1111111111111111111111111111110, -2 - 2**-29),
    (0x01, 1, 0b1111111111111111111111111111101, -2 - 2**-20 + 2**-30),
    (0x7F, 1, 0b0000000000000000000000000000001, (-2 + 2**-31) * 2**127),
    (0x7F, 1, 0b0000000000000000000000000000000, -(2**128)),
)

TEST_CASES = []
for e, s, m, val_out in TEST_FROM_COMPONENTS:
    val_in = np.uint64((e << 32) + (s << 31) + m)
    TEST_CASES.append((int(val_in), pytest.approx(val_out)))
SIZE = 40


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
class TestSpecificCases(SpecificCasesBase):
    def test_py_func(self, val_in, val_out):
        assert py_func(val_in) == val_out

    def test_c_func(self, val_in, val_out):
        assert c_func(val_in) == val_out

    # def test_py_ufunc(self, val_in, val_out):
    #     data = self.make_ndarray(val_in, SIZE)
    #     assert list(py_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_c_ufunc(self, val_in, val_out):
        print(f"{val_in:010x}")
        data = self.make_ndarray(val_in, SIZE)
        print(data.dtype, f"{data[0]:010x}")
        assert list(c_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_func(self, val_in, val_out):
        assert func(val_in) == val_out

    def test_ufunc(self, val_in, val_out):
        data = self.make_ndarray(val_in, SIZE)
        assert list(ufunc(data)) == [val_out] * self.ARRAY_SIZE
