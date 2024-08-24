import pytest
from typeconvert._c.func import bcd as c_func
from typeconvert._c.ufunc import bcd as c_ufunc
from typeconvert._py.func import bcd as py_func

# from typeconvert._py.ufunc import bcd as py_ufunc
from typeconvert.func import bcd as func
from typeconvert.ufunc import bcd as ufunc

from .conftest import NPY_CAST_SAFE, SpecificCasesBase

# https://pubs.usgs.gov/of/2005/1424/
TEST_CASES = [
    (0x03, 3),
    (0x12, 12),
    (0x1234, 1234),
    (0x1986, 1986),
    (0x12345678, 12345678),
    (0x19860101, 19860101),
    (0x20200501, 20200501),
    (0x1234567890123456, 1234567890123456),
    # Erroneous Values Test Cases
    (0x1A, -1),
    (0xA1, -1),
    (0xAA, -1),
    (0xFF, -1),
    (0x111A, -1),
    (0xA111, -1),
    (0xAAAA, -1),
    (0xFFFF, -1),
    (0x1111111A, -1),
    (0xA1111111, -1),
    (0xAAAAAAAA, -1),
    (0xFFFFFFFF, -1),
    (0x111111111111111A, -1),
    (0xA111111111111111, -1),
    (0xAAAAAAAAAAAAAAAA, -1),
    (0xFFFFFFFFFFFFFFFF, -1),
]


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
class TestSpecificCases(SpecificCasesBase):
    def test_py_func(self, val_in, val_out):
        assert py_func(val_in) == val_out

    def test_c_func(self, val_in, val_out):
        assert c_func(val_in) == val_out

    # @pytest.mark.xfail(NPY_CAST_SAFE, reason="numpy will not allow unsafe casting")
    # def test_py_ufunc(self, val_in, val_out):
    #     size = 4 * len(f"{val_in:x}")
    #     data = self.make_ndarray(val_in, size)
    #     assert list(py_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_c_ufunc(self, val_in, val_out):
        size = 4 * len(f"{val_in:x}")
        data = self.make_ndarray(val_in, size)
        print(f"{val_in:x}", c_ufunc(data).dtype)
        assert list(c_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_func(self, val_in, val_out):
        assert func(val_in) == val_out

    def test_ufunc(self, val_in, val_out):
        print(f"val_in = 0x{val_in:x}, val_out = {val_out}")
        size = 4 * len(f"{val_in:x}")
        data = self.make_ndarray(val_in, size)
        print(size, data.dtype)
        assert list(ufunc(data)) == [val_out] * self.ARRAY_SIZE
