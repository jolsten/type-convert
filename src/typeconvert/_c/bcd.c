#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>

/*
 * bcd_ufunc.c
 * This is the C code for a numpy ufunc converting an 
 * arbitrary-length (2-64 bits) unsigned integer to that
 * binary value's representation as a twos-complement number.
 * 
 * This method only works on systems which use arithmetic
 * right shift on negative signed integers. Which is,
 * hopefully, everywhere this gets used.
 */

/* The loop definition must precede the PyMODINIT_FUNC. */

static void uint8_bcd(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];
    uint8_t input, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint8_t *)in1;
        
        result = ((input & 0xF0) >> 4) * 10
               + ((input & 0x0F));

        *((uint8_t *)out1) = (uint8_t) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint16_bcd(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];
    uint16_t input, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint16_t *)in1;
        
        result = ((input & 0xF000) >> 12) * 1000
               + ((input & 0x0F00) >>  8) * 100
               + ((input & 0x00F0) >>  4) * 10
               + ((input & 0x000F));

        *((uint16_t *)out1) = (uint16_t) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint32_bcd(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];
    uint32_t input, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint32_t *)in1;
        
        result = ((input & 0xF0000000) >> 28) * 10000000
               + ((input & 0x0F000000) >> 24) * 1000000
               + ((input & 0x00F00000) >> 20) * 100000
               + ((input & 0x000F0000) >> 16) * 10000
               + ((input & 0x0000F000) >> 12) * 1000
               + ((input & 0x00000F00) >>  8) * 100
               + ((input & 0x000000F0) >>  4) * 10
               + ((input & 0x0000000F));


        *((uint32_t *)out1) = (uint32_t) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

static void uint64_bcd(char **args, const npy_intp *dimensions,
                             const npy_intp *steps, void *data)
{
    npy_intp i;
    npy_intp n = dimensions[0];
    char *in1 = args[0];
    char *out1 = args[1];
    npy_intp in1_step = steps[0];
    npy_intp out1_step = steps[1];
    uint64_t input, result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint64_t *)in1;
        
        result = ((input & 0xF000000000000000) >> 60) * 1000000000000000
               + ((input & 0x0F00000000000000) >> 56) * 100000000000000
               + ((input & 0x00F0000000000000) >> 52) * 10000000000000
               + ((input & 0x000F000000000000) >> 48) * 1000000000000
               + ((input & 0x0000F00000000000) >> 44) * 100000000000
               + ((input & 0x00000F0000000000) >> 40) * 10000000000
               + ((input & 0x000000F000000000) >> 36) * 1000000000
               + ((input & 0x0000000F00000000) >> 32) * 100000000
               + ((input & 0x00000000F0000000) >> 28) * 10000000
               + ((input & 0x000000000F000000) >> 24) * 1000000
               + ((input & 0x0000000000F00000) >> 20) * 100000
               + ((input & 0x00000000000F0000) >> 16) * 10000
               + ((input & 0x000000000000F000) >> 12) * 1000
               + ((input & 0x0000000000000F00) >>  8) * 100
               + ((input & 0x00000000000000F0) >>  4) * 10
               + ((input & 0x000000000000000F));

        *((uint64_t *)out1) = (uint64_t) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction bcd_funcs[4] = {&uint8_bcd, &uint16_bcd, &uint32_bcd, &uint64_bcd};

/* These are the input and return dtypes of ufunc.*/

static char bcd_types[8] = {
    NPY_UINT8, NPY_UINT8,
    NPY_UINT16, NPY_UINT16,
    NPY_UINT32, NPY_UINT32,
    NPY_UINT64, NPY_UINT64,
};