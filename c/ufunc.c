#include <Python.h>
#include "./onescomp.c"
#include "./twoscomp.c"

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

    return m;
}
