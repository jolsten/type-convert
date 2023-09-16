#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/*
 * ti40.c
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint64_ti40(char **args, const npy_intp *dimensions,
                        const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];

    uint64_t unsigned_int, m, e, s;
    double M, E, S, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        unsigned_int = *(uint64_t *)in1;

        e = (unsigned_int & 0xFF00000000) >> 32;
        s = (unsigned_int & 0x0080000000) >> 31;
        m = (unsigned_int & 0x007FFFFFFF);

        if (e == 0b10000000) {
            result = (double) 0.0f;
        } else {
            // Equivalent to (-2) ** s
            S = (s == 0) ? (double) 1.0f : (double) -2.0f;
            E = (double) twoscomp(e, 8);
            M = (double) m;
            result = (S + M/((double) (1UL << 31))) * pow((double) 2.0, E);
        }

        *((double *)out1) = (double) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction ti40_funcs[1] = {&uint64_ti40};

/* These are the input and return dtypes of ufunc.*/

static char ti40_types[2] = {
    NPY_UINT64, NPY_FLOAT64,
};