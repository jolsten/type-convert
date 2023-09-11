#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/*
 * twoscomp_ufunc.c
 * This is the C code for a numpy ufunc converting an 
 * arbitrary-length (2-64 bits) unsigned integer to that
 * binary value's representation as a twos-complement number.
 * 
 * This method only works on systems which use arithmetic
 * right shift on negative signed integers. Which is,
 * hopefully, everywhere this gets used.
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint32_ti32(char **args, const npy_intp *dimensions,
                           const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    uint32_t unsigned_int, m, e, s;
    double M, E, S, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        unsigned_int = *(uint32_t *)in1;

    // e = uint_to_twoscomp(
    //     (value & np.uint32(0xFF000000)) >> np.uint8(24), np.uint8(8)
    // )
    // s = (value & np.uint32(0x00800000)) >> np.uint8(23)
    // m = (value & np.uint32(0x007FFFFF))

    // if e == np.int64(-128):
    //     return np.float64(0)

    // S = np.float64(-2) ** s
    // E = np.float64(e)
    // M = np.float64(m)

    // return (S + M/np.float64(2**23)) * np.float64(2) ** E

        e = (unsigned_int & 0xFF000000) >> 24;
        s = (unsigned_int & 0x00800000) >> 23;
        m = (unsigned_int & 0x007FFFFF);

        if (e == -128) {
            result = 0.0f;
        } else {
            S = pow(-2.0, (double) s);
            E = (double) twoscomp(e, 8);
            M = (double) m;
            result = (S + M/((double) (1L << 23))) * pow((double) 2.0, E);
        }

        *((double *)out1) = (double) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction ti32_funcs[1] = {&uint32_ti32};

/* These are the input and return dtypes of ufunc.*/

static char ti32_types[2] = {
    NPY_UINT32, NPY_FLOAT64,
};