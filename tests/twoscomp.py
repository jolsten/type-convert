import pytest

TEST_CASES = {
    3: [
        (0b000,  0),
        (0b001,  1),
        (0b010,  2),
        (0b011,  3),
        (0b100, -4),
        (0b101, -3),
        (0b110, -2),
        (0b111, -1),
    ],

    8: [
        (0b00000000,    0),
        (0b00000001,    1),
        (0b00000010,    2),
        (0b01111110,  126),
        (0b01111111,  127),
        (0b10000000, -128),
        (0b10000001, -127),
        (0b10000010, -126),
        (0b11111110,   -2),
        (0b11111111,   -1),
    ]
}

tests = []
for size in TEST_CASES:
    for val_in, val_out in TEST_CASES[size]:
        tests.append( (size, val_in, val_out) )

from floatconvert.twoscomp import uint_to_twoscomp

@pytest.mark.parametrize('size, val_in, val_out', tests)
def test_twoscomp(size, val_in, val_out):
    assert uint_to_twoscomp(val_in, size) == val_out
