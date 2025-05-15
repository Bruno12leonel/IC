from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "mst.mst",
        ["mst/mst.pyx"],
        extra_compile_args=["-ffast-math"],
        include_dirs=[numpy.get_include()]
    )
]

setup(
    name='hdbscan_python',
    ext_modules=cythonize(extensions, annotate=True)
)
