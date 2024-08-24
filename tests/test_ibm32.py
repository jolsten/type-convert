import pytest
from typeconvert._c.func import ibm32 as c_func
from typeconvert._c.ufunc import ibm32 as c_ufunc
from typeconvert._py.func import ibm32 as py_func

# from typeconvert._py.ufunc import ibm32 as py_ufunc
from typeconvert.func import ibm32 as func
from typeconvert.ufunc import ibm32 as ufunc

from .conftest import SpecificCasesBase

# Reference:
# https://en.wikipedia.org/wiki/IBM_hexadecimal_floating-point

TEST_CASES = [
    (0x00100000, 5.397605e-79),
    (0x80100000, -5.397605e-79),
    (0x00000000, 0.0),
    (0x4019999A, 0.1),
    (0xC019999A, -0.1),
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
