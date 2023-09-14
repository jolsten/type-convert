#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <math.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"

/*
 * dec32.c
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

    // s = (value >> np.uint8(31)) * np.uint32(1)
    // e = (value >> np.uint8(23)) & np.uint32(0xFF)
    // m = (value & np.uint32(0x007FFFFF))

    // S = np.int8(-1) ** s
    // E = np.int16(e) - np.int16(128)
    // M = np.float64(m) / np.float64(2**24) + np.float64(0.5)

    // return np.float64(S * M * np.float64(2)**E)

static void uint64_dec32(char **args, const npy_intp *dimensions,
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

        s =  unsigned_int >> 31;
        e = (unsigned_int >> 23) & 0xFF;
        m = (unsigned_int & 0x007FFFFF);

        // Equivalent to (-1) ** s
        S = (s == 1) ? (double) -1.0f : (double) 1.0f;
        E = (double) ((signed int) e - 128);
        M = ((double) m) / ((double) (1UL << 24)) + (double) 0.5f;

        result = S * M * pow((double) 2.0f, E);

        *((double *)out1) = (double) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction dec32_funcs[1] = {&uint64_dec32};

/* These are the input and return dtypes of ufunc.*/

static char dec32_types[2] = {
    NPY_UINT32, NPY_FLOAT64,
};