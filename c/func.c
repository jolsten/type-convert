#include <Python.h>
#include <math.h>

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static signed long long twoscomp(unsigned long long uint,  unsigned char size) {
    unsigned char max_pos_val = 1;
    unsigned char pad_bits;
    union {
        unsigned long long int u;
        signed long long int s;
    } tmp;

    tmp.u = uint;

    // Determine max positive value for 2c int of this size
    max_pos_val = ((max_pos_val << (size-1))) - 1;

    // If there is a leading 0b1 (i.e. > max pos val),
    // then the number is negative, and the 2c value must
    // be obtained
    if (tmp.u > max_pos_val) {
        // Determine size of long long on this system,
        // to determine needed pad bits
        pad_bits = sizeof(tmp.u) * 8 - size;

        tmp.u = tmp.u << pad_bits;
        tmp.s = tmp.s >> pad_bits;
    }

    return tmp.s;
}

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
    float value, M, E;
    unsigned long m, e;

    /* Parse arguments */
    if(!PyArg_ParseTuple(args, "k", &unsigned_int)) {
        return NULL;
    }

    m = (unsigned_int & 0xFFFFFF00) >> 8;
    e = (unsigned_int & 0x000000FF);
    M = ((float) twoscomp(m, 24)) / ((float) (1 << 23));
    E = ((float) twoscomp(e,  8));
    value = M * pow(2.0f, E);

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