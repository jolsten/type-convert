import numpy as np
from setuptools import Extension, setup

extensions = [
    Extension("func", ["src/typeconvert/_c/func.c"], include_dirs=[np.get_include()]),
    Extension("ufunc", ["src/typeconvert/_c/ufunc.c"], include_dirs=[np.get_include()]),
]

setup(
    ext_package="typeconvert_ext",
    ext_modules=extensions,
)
