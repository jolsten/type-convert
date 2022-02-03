# type-converter
Convert various data types to signed integers or IEEE floating point.

# usage

For each conversion, there are three kinds of functions to import:
 - typeconvert.pyfunc: python functions
 - typeconvert.njit:   numba.njit'd functions
 - typeconvert.ufunc:  numba.vectorize'd functions (aka numpy ufuncs)

#### Arbitrary Size Two's-Complement

```python
In [1]: from typeconvert.pyfunc import twoscomp
In [2]: twoscomp(0x00, 8), twoscomp(0xFF, 8), twoscomp(0x7F, 8), twoscomp(0x80, 8)

Out[2]: (0, -1, 127, -128)
```

#### MIL-STD-1750A32

```python
In [3]: from typeconvert.pyfunc import milstd1750a32
In [4]: milstd1750a32(0x40000000), milstd1750a32(0x80000000), milstd1750a32(0x9FFFFF04)

Out[4]: (0.5, -1.0, -12.000002)
```
# functions

Choose what kind of function to import:
```python
from typeconvert.pyfunc import <function_name_here>

from typeconvert.njit import <function_name_here>

from typeconvert.ufunc import <function_name_here>
```

The signatures for the supported functions:
```python
onescomp(uint: np.uint64, size: int) -> np.int64
twoscomp(uint: np.uint64, size: int) -> np.int64

bcd(uint: np.uint64) -> np.uint64

milstd1750a32(uint: np.uint32) -> np.float32
milstd1750a48(uint: np.uint64) -> np.float64

ti32(uint: np.uint32) -> np.float32
ti40(uint: np.uint32) -> np.float64

dec32(uint: np.uint32) -> np.float64
dec64(uint: np.uint64) -> np.float64
dec64g(uint: np.uint64) -> np.float64

ibm32(uint: np.uint32) -> np.float64
ibm64(uint: np.uint64) -> np.float64
```
