from typing import Any, Optional
import numpy as np
from numpy.typing import DTypeLike
from typeconvert.utils import _bits_to_dtype

TEST_ARRAY_SIZE = 100


class SpecificCasesBase:
    WORD_SIZE = ...
    ARRAY_SIZE: int = 100

    py_func = ...
    py_ufunc = ...
    c_func = ...
    c_ufunc = ...

    def make_ndarray(self, val_in) -> np.ndarray:
        return np.array([val_in] * self.ARRAY_SIZE, dtype=self.dtype)

    @property
    def dtype(self) -> DTypeLike:
        return _bits_to_dtype(self.WORD_SIZE)

    def test_py_func(self, val_in, val_out):
        func = self.py_func
        func = staticmethod(func)
        assert func(val_in) == val_out

    def test_c_func(self, val_in, val_out):
        assert self.c_func(val_in) == val_out

    def test_py_ufunc(self, val_in, val_out):
        data = self.make_ndarray(val_in)
        result = self.py_ufunc(data)
        assert list(result) == [val_out] * self.ARRAY_SIZE

    def test_c_ufunc(self, val_in, val_out):
        data = self.make_ndarray(val_in)
        result = self.c_ufunc(data)
        assert list(result) == [val_out] * self.ARRAY_SIZE
