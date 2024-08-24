import pytest
from typeconvert._c.func import dec32 as c_func
from typeconvert._c.ufunc import dec32 as c_ufunc
from typeconvert._py.func import dec32 as py_func

# from typeconvert._py.ufunc import dec32 as py_ufunc
from typeconvert.func import dec32 as func
from typeconvert.ufunc import dec32 as ufunc

from .conftest import SpecificCasesBase

# https://pubs.usgs.gov/of/2005/1424/

TEST_CASES = [
    # F4
    (0x40800000, 1.000000),
    (0xC0800000, -1.000000),
    (0x41600000, 3.500000),
    (0xC1600000, -3.500000),
    (0x41490FD0, 3.141590),
    (0xC1490FD0, -3.141590),
    (0x7DF0BDC2, 9.9999999e36),
    (0xFDF0BDC2, -9.9999999e36),
    (0x03081CEA, 9.9999999e-38),
    (0x83081CEA, -9.9999999e-38),
    (0x409E0652, 1.234568),
    (0xC09E0652, -1.234568),
    (0x7FFFFFFF, 1.7014118e38),  # last two not from reference
    (0xFFFFFFFF, -1.7014118e38),
]
TEST_CASES = [(a, pytest.approx(b)) for a, b in TEST_CASES]
SIZE = 32


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
