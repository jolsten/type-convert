#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>

/*
 * milstd1750a48.c
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint64_milstd1750a48(char **args, const npy_intp *dimensions,
                                 const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    uint64_t unsigned_int, m, e;
    double M, E;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        unsigned_int = *(uint64_t *)in1;

        m = ((unsigned_int & 0xFFFFFF000000) >> 8)
          +  (unsigned_int & 0x00000000FFFF);
        e =  (unsigned_int & 0x000000FF0000) >> 16;

        M = ((double) twoscomp(m, 40)) / ((double) (1LL << 39));
        E = ((double) twoscomp(e,  8));

        *((double *)out1) = M * (double) pow(2.0f, E);
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction milstd1750a48_funcs[1] = {&uint64_milstd1750a48};

/* These are the input and return dtypes of ufunc.*/

static char milstd1750a48_types[2] = {
    NPY_UINT64, NPY_FLOAT64,
};