#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>

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

static void uint8_twoscomp(char **args, const npy_intp *dimensions,
                           const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];
    
    union {
        int8_t  s;
        uint8_t u;
    } tmp;
    uint8_t size, pad_bits;
    uint8_t max_pos_val;

    size = *(uint8_t *)in2;
    max_pos_val = ((uint8_t) (1 << (size-1))) - 1;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        tmp.u = *(uint8_t *)in1;

        if (tmp.u > max_pos_val) {
            pad_bits = 8 - size;
            tmp.u = tmp.u << pad_bits;
            tmp.s = tmp.s >> pad_bits;
        }

        *((int8_t *)out1) = (int8_t) tmp.s;
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

static void uint16_twoscomp(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];
    
    union {
        int16_t  s;
        uint16_t u;
    } tmp;
    uint8_t size, pad_bits;
    uint16_t max_pos_val;

    size = *(uint8_t *)in2;
    max_pos_val = ((uint16_t) (1 << (size-1))) - 1;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        tmp.u = *(uint16_t *)in1;

        if (tmp.u > max_pos_val) {
            pad_bits = 16 - size;
            tmp.u = tmp.u << pad_bits;
            tmp.s = tmp.s >> pad_bits;
        }

        *((int16_t *)out1) = (int16_t) tmp.s;
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

static void uint32_twoscomp(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];
    
    union {
        int32_t  s;
        uint32_t u;
    } tmp;
    uint8_t size, pad_bits;
    uint32_t max_pos_val;

    size = *(uint8_t *)in2;
    max_pos_val = ((uint32_t) (1 << (size-1))) - 1;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        tmp.u = *(uint32_t *)in1;

        if (tmp.u > max_pos_val) {
            pad_bits = 32 - size;
            tmp.u = tmp.u << pad_bits;
            tmp.s = tmp.s >> pad_bits;
        }

        *((int32_t *)out1) = (int32_t) tmp.s;
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

static void uint64_twoscomp(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0], *in2 = args[1];
    char *out1 = args[2];
    npy_intp in1_step = steps[0], in2_step = steps[1];
    npy_intp out1_step = steps[2];
    
    union {
        int64_t  s;
        uint64_t u;
    } tmp;
    uint8_t size, pad_bits;
    uint64_t max_pos_val;

    size = *(uint8_t *)in2;
    max_pos_val = ((uint64_t) (1 << (size-1))) - 1;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        tmp.u = *(uint64_t *)in1;

        if (tmp.u > max_pos_val) {
            pad_bits = 64 - size;
            tmp.u = tmp.u << pad_bits;
            tmp.s = tmp.s >> pad_bits;
        }

        *((int64_t *)out1) = (int64_t) tmp.s;
        /* END main ufunc computation */

        in1 += in1_step;
        in2 += in2_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction twoscomp_funcs[4] = {&uint8_twoscomp, &uint16_twoscomp, &uint32_twoscomp, &uint64_twoscomp};

/* These are the input and return dtypes of ufunc.*/

static char twoscomp_types[12] = {
    NPY_UINT8, NPY_UINT8, NPY_INT8,
    NPY_UINT16, NPY_UINT8, NPY_INT16,
    NPY_UINT32, NPY_UINT8, NPY_INT32,
    NPY_UINT64, NPY_UINT8, NPY_INT64,
};