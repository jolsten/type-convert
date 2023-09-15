# type-converter
Convert various unsigned integers to non-native Python or Numpy types.

# function types

For each conversion, there are two kinds of functions to import:
 - typeconvert.func:  scalar functions
 - typeconvert.ufunc: numpy universal functions (ufunc)

Choose what kind of function to import:
```python
from typeconvert.func import <function_name_here>
from typeconvert.ufunc import <function_name_here>
```

## scalar functions

The scalar functions take unsigned Python `int` as inputs and return either an `int` or Pytho `float` as appropriate.

## numpy universal functions

The numpy universal functions take the minimum-sized np.uint<> for the given type, and return the minimum-sized np.uint<> or np.float<> for the dynamic range of the output.



# examples

## Two's-Complement

```python
In [1]: from typeconvert.func import twoscomp
In [2]: twoscomp(0x00, 8), twoscomp(0xFF, 8), twoscomp(0x7F, 8), twoscomp(0x80, 8)

Out[2]: (0, -1, 127, -128)
```

## MIL-STD-1750A32

```python
In [3]: from typeconvert.func import milstd1750a32
In [4]: milstd1750a32(0x40000000), milstd1750a32(0x80000000), milstd1750a32(0x9FFFFF04)

Out[4]: (0.5, -1.0, -12.000002)
```
