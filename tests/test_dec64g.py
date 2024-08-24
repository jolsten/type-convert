import pytest
from typeconvert._c.func import dec64g as c_func
from typeconvert._c.ufunc import dec64g as c_ufunc
from typeconvert._py.func import dec64g as py_func

# from typeconvert._py.ufunc import dec64g as py_ufunc
from typeconvert.func import dec64g as func
from typeconvert.ufunc import dec64g as ufunc

from .conftest import SpecificCasesBase

TEST_CASES = [
    (0x4010000000000000, 1.000000000000000),
    (0xC010000000000000, -1.000000000000000),
    (0x402C000000000000, 3.500000000000000),
    (0xC02C000000000000, -3.500000000000000),
    (0x402921FB54442D18, 3.141592653589793),
    (0xC02921FB54442D18, -3.141592653589793),
    (0x47BE17B84357691B, 1.0000000000000000e37),
    (0xC7BE17B84357691B, -1.0000000000000000e37),
    (0x3861039D428A8B8F, 9.9999999999999999e-38),
    (0xB861039D428A8B8F, -9.9999999999999999e-38),
    (0x4013C0CA428C59DD, 1.234567890123450),
    (0xC013C0CA428C59DD, -1.234567890123450),
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
