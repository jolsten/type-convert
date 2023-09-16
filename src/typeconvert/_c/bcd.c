#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include <math.h>
#include <inttypes.h>

/*
 * bcd_ufunc.c
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
    uint8_t input, place;
    uint8_t digit;
    int8_t result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint8_t *)in1;
        
        result = 0;
        place = 1;
        while (input > 0) {
            digit = input & 0xF;
            if (digit >= 10) {
                result = -1;
                input = 0;
            } else {
                result += (int8_t) (digit * place);
                place *= 10;
                input = input >> 4;
            }
        }

        *((int8_t *)out1) = (int8_t) result;
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
    uint16_t input, place;
    uint8_t digit;
    int16_t result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint16_t *)in1;
        
        result = 0;
        place = 1;
        while (input > 0) {
            digit = input & 0xF;
            if (digit >= 10) {
                result = -1;
                input = 0;
            } else {
                result += (int16_t) (digit * place);
                place *= 10;
                input = input >> 4;
            }
        }

        *((int16_t *)out1) = (int16_t) result;
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
    uint32_t input, place;
    uint8_t digit;
    int32_t result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint32_t *)in1;
        
        result = 0;
        place = 1;
        while (input > 0) {
            digit = input & 0xF;
            if (digit >= 10) {
                result = -1;
                input = 0;
            } else {
                result += (int32_t) (digit * place);
                place *= 10;
                input = input >> 4;
            }
        }

        *((int32_t *)out1) = (int32_t) result;
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
    uint64_t input, place;
    uint8_t digit;
    int64_t result;

    for (i = 0; i < n; i++) {
        /* BEGIN main ufunc computation */
        input = *(uint64_t *)in1;
        
        result = 0;
        place = 1;
        while (input > 0) {
            digit = input & 0xF;
            if (digit >= 10) {
                result = -1;
                input = 0;
            } else {
                result += (int64_t) (digit * place);
                place *= 10;
                input = input >> 4;
            }
        }

        *((int64_t *)out1) = (int64_t) result;
        /* END main ufunc computation */

        in1 += in1_step;
        out1 += out1_step;
    }
}

/*This a pointer to the above function*/
PyUFuncGenericFunction bcd_funcs[4] = {&uint8_bcd, &uint16_bcd, &uint32_bcd, &uint64_bcd};

/* These are the input and return dtypes of ufunc.*/

static char bcd_types[8] = {
    NPY_UINT8, NPY_INT8,
    NPY_UINT16, NPY_INT16,
    NPY_UINT32, NPY_INT32,
    NPY_UINT64, NPY_INT64,
};