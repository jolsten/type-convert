import pathlib

import numpy
from setuptools import Extension, setup
from setuptools_scm.version import get_local_dirty_tag


def clean_scheme(version):
    return get_local_dirty_tag(version) if version.dirty else ""


npy_include = numpy.get_include()
typeconvert_src = pathlib.Path(__file__).parent / "src" / "typeconvert" / "_c"
includes = [npy_include, typeconvert_src]

extensions = [
    Extension(
        "typeconvert._c.func",
        sources=["src/typeconvert/_c/func.c"],
        include_dirs=includes,
    ),
    Extension(
        "typeconvert._c.ufunc",
        sources=["src/typeconvert/_c/ufunc.c"],
        include_dirs=includes,
    ),
]

setup(
    name="typeconvert",
    use_scm_version={"local_scheme": clean_scheme},
    ext_modules=extensions,
)
