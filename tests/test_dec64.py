import pytest
from typeconvert._c.func import dec64 as c_func
from typeconvert._c.ufunc import dec64 as c_ufunc
from typeconvert._py.func import dec64 as py_func

# from typeconvert._py.ufunc import dec64 as py_ufunc
from typeconvert.func import dec64 as func
from typeconvert.ufunc import dec64 as ufunc

from .conftest import SpecificCasesBase

TEST_CASES = [
    (0x4080000000000000, 1.000000000000000),
    (0xC080000000000000, -1.000000000000000),
    (0x4160000000000000, 3.500000000000000),
    (0xC160000000000000, -3.500000000000000),
    (0x41490FDAA22168BE, 3.141592653589793),
    (0xC1490FDAA22168BE, -3.141592653589793),
    (0x7DF0BDC21ABB48DB, 1.0000000000000000e37),
    (0xFDF0BDC21ABB48DB, -1.0000000000000000e37),
    (0x03081CEA14545C75, 9.9999999999999999e-38),
    (0x83081CEA14545C75, -9.9999999999999999e-38),
    (0x409E06521462CEE7, 1.234567890123450),
    (0xC09E06521462CEE7, -1.234567890123450),
]
TEST_CASES = [(a, pytest.approx(b)) for a, b in TEST_CASES]
SIZE = 64


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
        data = self.make_ndarray(val_in, SIZE)
        assert list(c_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_func(self, val_in, val_out):
        assert func(val_in) == val_out

    def test_ufunc(self, val_in, val_out):
        data = self.make_ndarray(val_in, SIZE)
        assert list(ufunc(data)) == [val_out] * self.ARRAY_SIZE
