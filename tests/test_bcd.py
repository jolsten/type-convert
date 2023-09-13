import pytest
from typeconvert._py.func import bcd as py_func
from typeconvert._py.ufunc import bcd as py_ufunc
from typeconvert._c.func import bcd as c_func
from typeconvert._c.ufunc import bcd as c_ufunc
from .conftest import SpecificCasesBase, NPY_CAST_SAFE

# https://pubs.usgs.gov/of/2005/1424/
TEST_CASES = [
    # 1,
    # 10,
    # 100,
    # 1_000,
    # 10_000,
    # 100_000,
    # 1_000_000,
    # 10_000_000,
    # 100_000_000,
    0x1986,
    0x19860101,
    0x20200501,
]
TEST_CASES = [(x, int(f"{x:x}")) for x in TEST_CASES]


# digits = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.uint8)
# np.random.seed(0)
# TEST_CASES_GEN = np.random.choice(digits, size=(100, 16))

# TEST_CASES = []
# for row in TEST_CASES_GEN:
#     hex_str = "".join([str(x) for x in row])
#     val_in = np.uint64(int(hex_str, base=16))
#     val_out = np.uint64(int(hex(val_in)[2:], base=10))
#     TEST_CASES.append((val_in, val_out))
# SIZE = 64

# [print(case) for case in TEST_CASES]


@pytest.mark.parametrize("val_in, val_out", TEST_CASES)
class TestSpecificCases(SpecificCasesBase):
    # def test_py_func(self, val_in, val_out):
    #     assert py_func(val_in) == val_out

    def test_c_func(self, val_in, val_out):
        assert c_func(val_in) == val_out

    @pytest.mark.skipif(NPY_CAST_SAFE, reason="numpy will not allow unsafe casting")
    def test_py_ufunc(self, val_in, val_out):
        size = 4 * len(str(val_in))
        data = self.make_ndarray(val_in, size)
        assert list(py_ufunc(data)) == [val_out] * self.ARRAY_SIZE

    def test_c_ufunc(self, val_in, val_out):
        size = 4 * len(str(val_in))
        data = self.make_ndarray(val_in, size)
        print(val_in, val_out, data[0], data.dtype)
        assert list(c_ufunc(data)) == [val_out] * self.ARRAY_SIZE
