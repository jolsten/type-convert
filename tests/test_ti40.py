import pytest
import numpy as np
from typeconvert.py.twoscomp import func as uint_to_twoscomp
from typeconvert.py.func import ti40 as py_func
from typeconvert.py.ufunc import ti40 as py_ufunc
from .conftest import SpecificCasesBase

TEST_ARRAY_SIZE = 100

S = [0, 1]
E = [0x7F, 0x7E, 0x01, 0xFF]
M = [0x007FFFFFFF, 0x007FFFFFFE, 0x007FFFFFFD, 0x0000000002, 0x0000000001, 0x0000000000]

TEST_CASES = []
for s in S:
    for e in E:
        for m in M:
            val_in = np.uint64((e << 32) | (s << 31) | m)
            val_out = ((-2) ** s + float(m) / 2**31) * float(2) ** uint_to_twoscomp(
                e, 8
            )
            TEST_CASES.append((val_in, pytest.approx(val_out)))
SIZE = 40


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
