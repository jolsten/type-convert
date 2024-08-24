import pytest
from typeconvert._c.func import milstd1750a32 as c_func
from typeconvert._c.ufunc import milstd1750a32 as c_ufunc
from typeconvert._py.func import milstd1750a32 as py_func

# from typeconvert._py.ufunc import milstd1750a32 as py_ufunc
from typeconvert.func import milstd1750a32 as func
from typeconvert.ufunc import milstd1750a32 as ufunc

from .conftest import SpecificCasesBase

TEST_CASES = [
    (0x7FFFFF7F, 0.9999998 * 2**127),
    (0x4000007F, 0.5 * 2**127),
    (0x50000004, 0.625 * 2**4),
    (0x40000001, 0.5 * 2**1),
    (0x40000000, 0.5 * 2**0),
    (0x400000FF, 0.5 * 2**-1),
    (0x40000080, 0.5 * 2**-128),
    (0x00000000, 0.0 * 2**0),
    (0x80000000, -1.0 * 2**0),
    (0xBFFFFF80, -0.5000001 * 2**-128),
    (0x9FFFFF04, -0.7500001 * 2**4),
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
