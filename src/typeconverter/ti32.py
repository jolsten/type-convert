import numpy as np
from numba import njit, vectorize

signatures = [
    'f8(u4)',
]

from typeconverter.twoscomp import jfunc as uint_to_twoscomp

import struct

# def func(value: np.uint32) -> np.float64:
#     # Reference:
#     # https://stackoverflow.com/questions/64687130/convert-ti-tms320c30-32-bits-float-to-ieee-float-in-python
#     value = np.uint32(value)
#
#     frac = value & np.uint32(0x7FFFFF)
#     sign = (value >> np.uint8(23)) & np.uint32(1)
#     expo = (value >> np.uint8(24)) & np.uint32(0xFF)
#
#     if expo == np.uint32(0x80):
#         # Zero or Implied zero
#         return np.float64(0.0)
#     else:
#         # Add the IEEE exponent bias of 127
#         expo = (expo + np.uint32(127)) & np.uint32(0xFF)
#
#         if sign:
#             # Complement fraction
#             frac = np.uint32(0x00800000) - frac
#
#             # Propagate fraction overflow to exponent
#             expo = expo + (frac >> np.uint8(23))
#
#             # Clear potential overflow
#             frac = frac & np.uint32(0x007FFFFF)
#
#         if expo == np.uint32(0):
#             # Make implicit integer bit explicit
#             frac = frac + np.uint32(0x00800000)
#
#             # Denormalize, round to nearest-or-even
#             frac = (frac >> np.uint8(1)) + ((frac * np.uint32(3)) == np.uint32(3))
#
#         result = np.uint32(
#             (sign << np.uint8(31)) | (expo << np.uint8(23)) | frac
#         )
#
#         return result.view(np.float32)

def func(value: np.uint32) -> np.float64:
    # Reference:
    # https://www.ti.com/lit/an/spra400/spra400.pdf
    value = np.uint32(value)

    e = uint_to_twoscomp((value & np.uint32(0xFF000000)) >> np.uint8(24), np.uint8(8))
    s = (value & np.uint32(0x00800000)) >> np.uint8(23)
    m = (value & np.uint32(0x007FFFFF))

    if e == np.int64(-128):
        return np.float64(0)

    S = np.float64(-2) ** s
    E = np.float64(e)
    M = np.float64(m)

    return ( S + M/np.float64(2**23) ) * np.float64(2) ** E

jfunc = njit(signatures)(func)
ufunc = vectorize(signatures)(func)
