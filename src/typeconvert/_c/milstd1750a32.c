#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>

/*
 * milstd1750a32.c
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint32_milstd1750a32(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    uint32_t unsigned_int, m, e;
    float M, E;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        unsigned_int = *(uint32_t *)in1;

        m = (unsigned_int & 0xFFFFFF00) >> 8;
        e = (unsigned_int & 0x000000FF);

        M = ((float) twoscomp(m, 24)) / ((float) (1L << 23));
        E = ((float) twoscomp(e,  8));

        *((float *)out1) = M * (float) pow(2.0f, E);
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction milstd1750a32_funcs[1] = {&uint32_milstd1750a32};

/* These are the input and return dtypes of ufunc.*/

static char milstd1750a32_types[2] = {
    NPY_UINT32, NPY_FLOAT32,
};