#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/*
 * ibm32.c
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint64_ibm32(char **args, const npy_intp *dimensions,
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

        s = (unsigned_int & 0x80000000) >> 31;
        e = (unsigned_int & 0x7F000000) >> 24;
        m = (unsigned_int & 0x00FFFFFF);

        // Equivalent to (-1) ** s
        S = (s == 1) ? (double) -1.0f : (double) 1.0f;
        E = (double) ((signed int) e - 64);
        M = ((double) m) / ((double) (1U << 24));

        result = S * M * pow((double) 16.0f, E);

        *((double *)out1) = (double) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction ibm32_funcs[1] = {&uint64_ibm32};

/* These are the input and return dtypes of ufunc.*/

static char ibm32_types[2] = {
    NPY_UINT32, NPY_FLOAT64,
};