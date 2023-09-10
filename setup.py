from setuptools import Extension, setup
import numpy as np

extensions = [
    Extension("func", ["c/func.c"], include_dirs=[np.get_include()]),
    Extension("ufunc", ["c/ufunc.c"], include_dirs=[np.get_include()]),
]

setup(
    ext_package="typeconvert_ext",
    ext_modules=extensions,
)
