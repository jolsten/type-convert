import pytest
from typeconvert._c.func import onescomp as c_func
from typeconvert._c.ufunc import onescomp as c_ufunc
from typeconvert._py.func import onescomp as py_func

# from typeconvert._py.ufunc import onescomp as py_ufunc
from typeconvert.func import onescomp as func
from typeconvert.ufunc import onescomp as ufunc

from .conftest import SpecificCasesBase

TEST_CASES = {
    3: [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -3),
        (0b101, -2),
        (0b110, -1),
        (0b111, -0),
    ],
    8: [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -127),
        (0b10000001, -126),
        (0b10000010, -125),
        (0b11111110, -1),
        (0b11111111, -0),
    ],
    16: [
        (0x0000, 0),
        (0x7FFF, 2**15 - 1),
        (0x8000, -(2**15) + 1),
        (0xFFFF, 0),
    ],
    24: [
        (0x000000, 0),
        (0x7FFFFF, 2**23 - 1),
        (0x800000, -(2**23) + 1),
        (0xFFFFFF, 0),
    ],
    32: [
        (0x00000000, 0),
        (0x7FFFFFFF, 2**31 - 1),
        (0x80000000, -(2**31) + 1),
        (0xFFFFFFFF, 0),
    ],
    48: [
        (0x000000000000, 0),
        (0x7FFFFFFFFFFF, 2**47 - 1),
        (0x800000000000, -(2**47) + 1),
        (0xFFFFFFFFFFFF, 0),
    ],
    64: [
        (0x0000000000000000, 0),
        (0x7FFFFFFFFFFFFFFF, 2**63 - 1),
        (0x8000000000000000, -(2**63) + 1),
        (0xFFFFFFFFFFFFFFFF, 0),
    ],
}

tests = []
for size in TEST_CASES:
    for val_in, val_out in TEST_CASES[size]:
        tests.append((size, val_in, val_out))

# Min length = 2
# Max length = 64
for size in range(2, 65):
    tests.append((size, 0, 0))  # positive zero
    tests.append((size, 2**size - 1, 0))  # negative zero
    tests.append((size, 1, 1))  # min positive
    tests.append((size, 2 ** (size - 1) - 1, 2 ** (size - 1) - 1))  # max positive
    tests.append((size, 2**size - 2, -1))  # min negative
    tests.append((size, 1 << (size - 1), -(2 ** (size - 1)) + 1))  # max negative


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
