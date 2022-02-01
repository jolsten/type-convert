import pytest

from typeconverter.utils import bits_to_wordsize, mask


@pytest.mark.parametrize('size, value',
    (
        (1,1),
        (3,7),
        (4,0xF),
        (8,0xFF),
        (16,0xFFFF),
        (32,0xFFFFFFFF),
        (64,0xFFFFFFFFFFFFFFFF),
    ),
)
def test_mask(size, value):
    assert mask(size) == value
