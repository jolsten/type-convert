import pytest
import numpy as np
from typeconvert.py.func import bcd as py_func
from typeconvert.py.ufunc import bcd as py_ufunc
from .conftest import SpecificCasesBase

# https://pubs.usgs.gov/of/2005/1424/
digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
np.random.seed(0)
TEST_CASES_GEN = np.random.choice(digits, size=(100, 16))

TEST_CASES = []
for row in TEST_CASES_GEN:
    hex_str = "".join([str(x) for x in row])
    val_in = np.uint64(int(hex_str, base=16))
    val_out = np.uint64(int(hex(val_in)[2:], base=10))
    TEST_CASES.append((val_in, val_out))
SIZE = 64


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
class TestSpecificCases(SpecificCasesBase):
    def test_py_func(self, val_in, val_out):
        assert py_func(val_in) == val_out

    # def test_c_func(self, val_in, val_out):
    #     assert c_func(val_in) == val_out

    def test_py_ufunc(self, val_in, val_out):
        data = self.make_ndarray(val_in, SIZE)
        assert list(py_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    # def test_c_ufunc(self, val_in, val_out):
    #     data = self.make_ndarray(val_in, SIZE)
    #     assert list(c_ufunc(data)) == [val_out] * self.ARRAY_SIZE
