#include <Python.h>
#include <math.h>
#include "./utils.c"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION


static PyObject *method_onescomp(PyObject *self, PyObject *args) {
    unsigned long long tmp;
    unsigned char size;
    signed long long result;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "Kb", &tmp, &size)) {
        return NULL;
    }

    result = twoscomp(tmp, size);
    if (result < 0) {
        result += 1;
    }

    return PyLong_FromLongLong(result);
}

static PyObject *method_twoscomp(PyObject *self, PyObject *args) {
    unsigned long long tmp;
    unsigned char size;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "Kb", &tmp, &size)) {
        return NULL;
    }

    return PyLong_FromLongLong(twoscomp(tmp, size));
}

static PyObject *method_1750a32(PyObject *self, PyObject *args) {
    unsigned long input;
    uint32_t unsigned_int, m, e;
    double value, M, E;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &input)) {
        return NULL;
    }

    unsigned_int = (uint32_t) input;

    m = (unsigned_int & 0xFFFFFF00) >> 8;
    e = (unsigned_int & 0x000000FF);

    M = ((double) twoscomp(m, 24)) / ((double) (1ULL << 23));
    E = ((double) twoscomp(e,  8));
    value = (double) M * pow((double) 2.0f, E);

    return PyFloat_FromDouble(value);
}

static PyObject *method_1750a48(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e;
    double value, M, E;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    unsigned_int = (uint64_t) input;

    m = ((unsigned_int & 0xFFFFFF000000) >> 8)
      +  (unsigned_int & 0x00000000FFFF);
    e =  (unsigned_int & 0x000000FF0000) >> 16;

    M = ((double) twoscomp(m, 40)) / ((double) (1ULL << 39));
    E = ((double) twoscomp(e,  8));

    value = M * pow((double) 2.0, E);
    return PyFloat_FromDouble(value);
}

static PyObject *method_ti32(PyObject *self, PyObject *args) {
    unsigned long input;
    uint32_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &input)) {
        return NULL;
    }

    unsigned_int = (uint32_t) input;

    e = (unsigned_int & 0xFF000000) >> 24;
    s = (unsigned_int & 0x00800000) >> 23;
    m = (unsigned_int & 0x007FFFFF);

    if (e == 0b10000000) {
        result = (double) 0.0f;
    } else {
        // Equivalent to (-2) ** s
        S = (s == 0) ? (double) 1.0f : (double) -2.0f;
        E = (double) twoscomp(e, 8);
        M = (double) m;
        result = (S + M/((double) (1UL << 23))) * pow((double) 2.0, E);
    }

    return PyFloat_FromDouble(result);
}

static PyObject *method_ti40(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    unsigned_int = (uint64_t) input;

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

    return PyFloat_FromDouble(result);
}

static PyObject *method_ibm32(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &input)) {
        return NULL;
    }

    unsigned_int = (uint32_t) input;

    s = (unsigned_int & 0x80000000) >> 31;
    e = (unsigned_int & 0x7F000000) >> 24;
    m = (unsigned_int & 0x00FFFFFF);

    // Equivalent to (-1) ** s
    S = (s == 1) ? (double) -1.0f : (double) 1.0f;
    E = (double) ((signed int) e - 64);
    M = ((double) m) / ((double) (1U << 24));

    result = S * M * pow((double) 16.0f, E);

    return PyFloat_FromDouble(result);
}

static PyObject *method_ibm64(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    unsigned_int = (uint64_t) input;

    s = (unsigned_int & 0x8000000000000000) >> 63;
    e = (unsigned_int & 0x7F00000000000000) >> 56;
    m = (unsigned_int & 0x00FFFFFFFFFFFFFF);

    // Equivalent to (-1) ** s
    S = (s == 1) ? (double) -1.0f : (double) 1.0f;
    E = (double) ((signed int) e - 64);
    M = ((double) m) / ((double) (1ULL << 56));

    result = S * M * pow((double) 16.0f, E);

    return PyFloat_FromDouble(result);
}

static PyObject *method_dec32(PyObject *self, PyObject *args) {
    unsigned long input;
    uint32_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &input)) {
        return NULL;
    }

    unsigned_int = (uint32_t) input;

    s =  unsigned_int >> 31;
    e = (unsigned_int >> 23) & 0xFF;
    m = (unsigned_int & 0x007FFFFF);

    // Equivalent to (-1) ** s
    S = (s == 1) ? (double) -1.0f : (double) 1.0f;
    E = (double) ((signed int) e - 128);
    M = ((double) m) / ((double) (1UL << 24)) + (double) 0.5f;

    result = S * M * pow((double) 2.0f, E);

    return PyFloat_FromDouble(result);
}

static PyObject *method_dec64(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    unsigned_int = (uint64_t) input;

    s =  unsigned_int >> 63;
    e = (unsigned_int >> 55) & 0xFF;
    m = (unsigned_int & 0x007FFFFFFFFFFFFF);

    // Equivalent to (-1) ** s
    S = (s == 1) ? (double) -1.0f : (double) 1.0f;
    E = (double) ((signed int) e - 128);
    M = ((double) m) / ((double) (1ULL << 56)) + (double) 0.5f;

    result = S * M * pow((double) 2.0f, E);

    return PyFloat_FromDouble(result);
}

static PyObject *method_dec64g(PyObject *self, PyObject *args) {
    unsigned long long input;
    uint64_t unsigned_int, m, e, s;
    double result, M, E, S;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    unsigned_int = (uint64_t) input;

    s =  unsigned_int >> 63;
    e = (unsigned_int >> 52) & 0x7FF;
    m = (unsigned_int & 0x000FFFFFFFFFFFFF);

    // Equivalent to (-1) ** s
    S = (s == 1) ? (double) -1.0f : (double) 1.0f;
    E = (double) ((signed int) e - 1024);
    M = ((double) m) / ((double) (1ULL << 53)) + (double) 0.5f;

    result = S * M * pow((double) 2.0f, E);

    return PyFloat_FromDouble(result);
}

static PyObject *method_bcd(PyObject *self, PyObject *args) {
    unsigned long long input, place;
    unsigned char digit;
    signed long long result;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &input)) {
        return NULL;
    }

    result = 0;
    place = 1;
    while (input > 0) {
        digit = input & 0xF;
        if (digit >= 10) {
            return PyLong_FromLongLong((signed long long) -1);
            // result = -1;
            // input = 0;
        } else {
            result += (signed long long) (digit * place);
            place *= 10;
            input = input >> 4;
        }
    }

    return PyLong_FromLongLong(result);
}

static PyMethodDef TypeConversionMethods[] = {
    {"onescomp", method_onescomp, METH_VARARGS, "Python interface for the onescomp function"},
    {"twoscomp", method_twoscomp, METH_VARARGS, "Python interface for the twoscomp function"},
    {"milstd1750a32", method_1750a32, METH_VARARGS, "Python interface for the milstd1750a32 function"},
    {"milstd1750a48", method_1750a48, METH_VARARGS, "Python interface for the milstd1750a48 function"},
    {"ti32", method_ti32, METH_VARARGS, "Python interface for the ti32 function"},
    {"ti40", method_ti40, METH_VARARGS, "Python interface for the ti40 function"},
    {"ibm32", method_ibm32, METH_VARARGS, "Python interface for the ibm32 function"},
    {"ibm64", method_ibm64, METH_VARARGS, "Python interface for the ibm64 function"},
    {"dec32", method_dec32, METH_VARARGS, "Python interface for the dec32 function"},
    {"dec64", method_dec64, METH_VARARGS, "Python interface for the dec64 function"},
    {"dec64g", method_dec64g, METH_VARARGS, "Python interface for the dec64g function"},
    {"bcd", method_bcd, METH_VARARGS, "Python interface for the bcd function"},
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef func_module = {
    PyModuleDef_HEAD_INIT,
    "func",
    "Python interface for the various type interpretation functions",
    -1,
    TypeConversionMethods
};

PyMODINIT_FUNC PyInit_func(void) {
    return PyModule_Create(&func_module);
}