import pytest
import numpy as np
from typeconvert.utils import bits_to_wordsize
from typeconvert.func import onescomp as func
from typeconvert.ufunc import onescomp as ufunc

# Min length = 2
# Max length = 64
ALL_SIZES = [x for x in range(2, 65)]

TEST_ARRAY_SIZE = 100
TEST_CASES = {
    3: [
        (0b000, 0),
        (0b001, 1),
        (0b010, 2),
        (0b011, 3),
        (0b100, -3),
        (0b101, -2),
        (0b110, -1),
        (0b111, -0),
    ],
    8: [
        (0b00000000, 0),
        (0b00000001, 1),
        (0b00000010, 2),
        (0b01111110, 126),
        (0b01111111, 127),
        (0b10000000, -127),
        (0b10000001, -126),
        (0b10000010, -125),
        (0b11111110, -1),
        (0b11111111, -0),
    ],
    16: [
        (0x0000, 0),
        (0x7FFF, 2**15 - 1),
        (0x8000, -(2**15) + 1),
        (0xFFFF, 0),
    ],
    24: [
        (0x000000, 0),
        (0x7FFFFF, 2**23 - 1),
        (0x800000, -(2**23) + 1),
        (0xFFFFFF, 0),
    ],
    32: [
        (0x00000000, 0),
        (0x7FFFFFFF, 2**31 - 1),
        (0x80000000, -(2**31) + 1),
        (0xFFFFFFFF, 0),
    ],
    48: [
        (0x000000000000, 0),
        (0x7FFFFFFFFFFF, 2**47 - 1),
        (0x800000000000, -(2**47) + 1),
        (0xFFFFFFFFFFFF, 0),
    ],
    64: [
        (0x0000000000000000, 0),
        (0x7FFFFFFFFFFFFFFF, 2**63 - 1),
        (0x8000000000000000, -(2**63) + 1),
        (0xFFFFFFFFFFFFFFFF, 0),
    ],
}

tests = []
for size in TEST_CASES:
    for val_in, val_out in TEST_CASES[size]:
        tests.append((size, val_in, val_out))


@pytest.mark.parametrize("size, val_in, val_out", tests)
def test_func_specific_cases(size, val_in, val_out):
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_pos_zero(size):
    val_in, val_out = 0, 0
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_neg_zero(size):
    val_in, val_out = 2**size - 1, -0
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_min_positive(size):
    val_in, val_out = 1, 1
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_max_positive(size):
    val_in = 2 ** (size - 1) - 1
    val_out = val_in
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_min_negative(size):
    val_in, val_out = 2**size - 2, -1
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size", ALL_SIZES)
def test_func_max_negative(size):
    val_in = 1 << (size - 1)
    val_out = -(2 ** (size - 1)) + 1
    assert func(val_in, size) == val_out


@pytest.mark.parametrize("size, val_in, val_out", tests)
def test_ufunc_specific_cases(size, val_in, val_out):
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    print(ufunc(data, size)[0])
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_pos_zero(size):
    val_in, val_out = 0, 0
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_neg_zero(size):
    val_in, val_out = 2**size - 1, -0
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_min_positive(size):
    val_in, val_out = 1, 1
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_max_positive(size):
    val_in = 2 ** (size - 1) - 1
    val_out = val_in
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_min_negative(size):
    val_in, val_out = 2**size - 2, -1
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE


@pytest.mark.parametrize("size", ALL_SIZES)
def test_ufunc_max_negative(size):
    val_in = 1 << (size - 1)
    val_out = -(2 ** (size - 1)) + 1
    data = np.array([val_in] * TEST_ARRAY_SIZE, dtype=f"uint{bits_to_wordsize(size)}")
    assert list(ufunc(data, size)) == [val_out] * TEST_ARRAY_SIZE
