from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import os.path as op

import numpy as np

def readlines(f):
    with open(f) as fh:
        return fh.readlines()

extensions = [
    Extension("mujoco_py_deriv",
              ["mujoco_py_deriv.pyx",
               "mujoco_deriv_struct.c"],
              include_dirs=[
                  np.get_include(),
                  "{home}/.mujoco/mujoco200/include/".format(home=op.expanduser("~"))],
              library_dirs=["{home}/.mujoco/mujoco200/bin/".format(home=op.expanduser("~"))],
              extra_compile_args=['-fopenmp'],
              libraries=["mujoco200", "glew", "GL", "gomp", "m"]),
]

setup(
    name = 'mujoco_py_deriv',
    version = '0.1',
    ext_modules = cythonize(extensions),
    package_data = {
        '': ['*.xml', '*.stl', '*.so', '*.pyd'],
    },
    setup_requires=readlines('pip-requirements.txt')
)

