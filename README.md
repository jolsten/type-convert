# type-converter
Convert various data types to signed integers or IEEE floating point.

# usage

## Arbitrary Size Two's-Complement

In [1]: from typeconverter.pyfunc import twoscomp
In [2]: twoscomp(0x00, 8), twoscomp(0xFF, 8), twoscomp(0x7F, 8), twoscomp(0x80, 8)

Out[2]: (0, -1, 127, -128)

## MIL-STD-1750A32

In [3]: from typeconverter.pyfunc import milstd1750a32
In [4]: milstd1750a32(0x40000000), milstd1750a32(0x80000000), milstd1750a32(0x9FFFFF04)

Out[4]: (0.5, -1.0, -12.000002)
