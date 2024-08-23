import numpy as np
from typeconvert.utils import NPY_CAST_SAFE, _bits_to_dtype


class SpecificCasesBase:
    ARRAY_SIZE: int = 2

    def make_ndarray(self, val_in, size) -> np.ndarray:
        return np.array([val_in] * self.ARRAY_SIZE, dtype=_bits_to_dtype(size))
