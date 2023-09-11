#include <Python.h>
#include "./utils.c"
#include "./onescomp.c"
#include "./twoscomp.c"
#include "./milstd1750a32.c"
#include "./milstd1750a48.c"
#include "./ti32.c"

#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION

static PyMethodDef Methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "ufunc",
    NULL,
    -1,
    Methods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_ufunc(void)
{
    PyObject *m, *d;

    import_array();
    import_umath();

    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    d = PyModule_GetDict(m);

    PyObject *onescomp = PyUFunc_FromFuncAndData(
        onescomp_funcs, NULL, onescomp_types, 4, 2, 1, PyUFunc_None, "onescomp", "ufunc_docstring", 0
    );
    PyDict_SetItemString(d, "onescomp", onescomp);
    Py_DECREF(onescomp);

    PyObject *twoscomp = PyUFunc_FromFuncAndData(
        twoscomp_funcs, NULL, twoscomp_types, 4, 2, 1, PyUFunc_None, "twoscomp", "ufunc_docstring", 0
    );
    PyDict_SetItemString(d, "twoscomp", twoscomp);
    Py_DECREF(twoscomp);

    PyObject *milstd1750a32 = PyUFunc_FromFuncAndData(
        milstd1750a32_funcs, NULL, milstd1750a32_types, 1, 1, 1, PyUFunc_None, "milstd1750a32", "ufunc_docstring", 0
    );
    PyDict_SetItemString(d, "milstd1750a32", milstd1750a32);
    Py_DECREF(milstd1750a32);

    PyObject *milstd1750a48 = PyUFunc_FromFuncAndData(
        milstd1750a48_funcs, NULL, milstd1750a48_types, 1, 1, 1, PyUFunc_None, "milstd1750a48", "ufunc_docstring", 0
    );
    PyDict_SetItemString(d, "milstd1750a48", milstd1750a48);
    Py_DECREF(milstd1750a48);

    PyObject *ti32 = PyUFunc_FromFuncAndData(
        ti32_funcs, NULL, ti32_types, 1, 1, 1, PyUFunc_None, "ti32", "ufunc_docstring", 0
    );
    PyDict_SetItemString(d, "ti32", ti32);
    Py_DECREF(ti32);

    return m;
}
