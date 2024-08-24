import pytest
from typeconvert._c.func import twoscomp as c_func
from typeconvert._c.ufunc import twoscomp as c_ufunc
from typeconvert._py.func import twoscomp as py_func

# from typeconvert._py.ufunc import twoscomp as py_ufunc
from typeconvert.func import twoscomp as func
from typeconvert.ufunc import twoscomp as ufunc

from .conftest import SpecificCasesBase

TEST_CASES = {
    3: [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ],
    8: [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -128),
        (0b10000001, -127),
        (0b10000010, -126),
        (0b11111110, -2),
        (0b11111111, -1),
    ],
    16: [
        (0x0000, 0),
        (0x7FFF, 2**15 - 1),
        (0x8000, -(2**15)),
        (0xFFFF, -1),
    ],
    24: [
        (0x000000, 0),
        (0x7FFFFF, 2**23 - 1),
        (0x800000, -(2**23)),
        (0xFFFFFF, -1),
    ],
    32: [
        (0x00000000, 0),
        (0x7FFFFFFF, 2**31 - 1),
        (0x80000000, -(2**31)),
        (0xFFFFFFFF, -1),
    ],
    48: [
        (0x000000000000, 0),
        (0x7FFFFFFFFFFF, 2**47 - 1),
        (0x800000000000, -(2**47)),
        (0xFFFFFFFFFFFF, -1),
    ],
    64: [
        (0x0000000000000000, 0),
        (0x7FFFFFFFFFFFFFFF, 2**63 - 1),
        (0x8000000000000000, -(2**63)),
        (0xFFFFFFFFFFFFFFFF, -1),
    ],
}

tests = []
for size in TEST_CASES:
    for val_in, val_out in TEST_CASES[size]:
        tests.append((size, val_in, val_out))

# Min length = 2
# Max length = 64
for size in range(2, 65):
    tests.append((size, 0, 0))  # zero
    tests.append((size, 1, 1))  # min positive
    tests.append((size, 2 ** (size - 1) - 1, 2 ** (size - 1) - 1))  # max positive
    tests.append((size, 2**size - 1, -1))  # min negative
    tests.append((size, 1 << (size - 1), -(2 ** (size - 1))))  # max negative


@pytest.mark.parametrize("size, val_in, val_out", tests)
class TestSpecificCases(SpecificCasesBase):
    def test_py_func(self, size, val_in, val_out):
        assert py_func(val_in, size) == val_out

    def test_c_func(self, size, val_in, val_out):
        assert c_func(val_in, size) == val_out

    # def test_py_ufunc(self, size, val_in, val_out):
    #     data = self.make_ndarray(val_in, size)
    #     assert list(py_ufunc(data, size)) == [val_out] * self.ARRAY_SIZE

    def test_c_ufunc(self, size, val_in, val_out):
        data = self.make_ndarray(val_in, size)
        assert list(c_ufunc(data, size)) == [val_out] * self.ARRAY_SIZE

    def test_func(self, size, val_in, val_out):
        assert func(val_in, size) == val_out

    def test_ufunc(self, size, val_in, val_out):
        data = self.make_ndarray(val_in, size)
        assert list(ufunc(data, size)) == [val_out] * self.ARRAY_SIZE
