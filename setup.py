from setuptools import Extension, setup
import numpy as np

# extensions = [Extension("typeconvert.twoscomp", ["typeconvert/twoscomp" + ext])]
extensions = [
    Extension("ufunc.twoscomp", ["c/twoscomp.c"], include_dirs=[np.get_include()])
]

setup(
    ext_package="typeconvert_extensions",
    ext_modules=extensions,
    # include_dirs=[np.get_include()],
)
