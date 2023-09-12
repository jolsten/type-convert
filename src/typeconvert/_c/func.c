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
    uint32_t unsigned_int;
    double value, M, E;
    unsigned long m, e;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &unsigned_int)) {
        return NULL;
    }

    m = (unsigned_int & 0xFFFFFF00) >> 8;
    e = (unsigned_int & 0x000000FF);
    M = ((double) twoscomp(m, 24)) / ((double) (1 << 23));
    E = ((double) twoscomp(e,  8));
    value = (double) M * pow(2.0f, E);

    return PyFloat_FromDouble((double) value);
}

static PyObject *method_1750a48(PyObject *self, PyObject *args) {
    unsigned long long unsigned_int, m, e;
    double value, M, E;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "K", &unsigned_int)) {
        return NULL;
    }

    m = ((unsigned_int & 0xFFFFFF000000) >> 8)
      +  (unsigned_int & 0x00000000FFFF);
    e =  (unsigned_int & 0x000000FF0000) >> 16;

    M = ((double) twoscomp(m, 40)) / ((double) (1LL << 39));
    E = ((double) twoscomp(e,  8));

    value = M * pow((double) 2.0, E);
    return PyFloat_FromDouble(value);
}

static PyMethodDef TypeConversionMethods[] = {
    {"onescomp", method_onescomp, METH_VARARGS, "Python interface for the onescomp function"},
    {"twoscomp", method_twoscomp, METH_VARARGS, "Python interface for the twoscomp function"},
    {"milstd1750a32", method_1750a32, METH_VARARGS, "Python interface for the milstd1750a32 function"},
    {"milstd1750a48", method_1750a48, METH_VARARGS, "Python interface for the milstd1750a48 function"},
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