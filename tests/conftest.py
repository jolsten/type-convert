import numpy as np
from typeconvert.utils import _bits_to_dtype, NPY_CAST_SAFE


class SpecificCasesBase:
    ARRAY_SIZE: int = 100

    def make_ndarray(self, val_in, size) -> np.ndarray:
        return np.array([val_in] * self.ARRAY_SIZE, dtype=_bits_to_dtype(size))
