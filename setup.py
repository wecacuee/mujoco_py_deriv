from setuptools import setup, find_packages
from setuptools.extension import Extension
from Cython.Build import cythonize

import os.path as op

import numpy as np

def readlines(f):
    with open(f, encoding='utf-8') as fh:
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
    version = '0.1.2',
    ext_modules = cythonize(extensions),
    package_data = {
        '': ['*.xml', '*.stl', '*.so', '*.pyd', '*.pyx'],
    },
    setup_requires=readlines('pip-requirements.txt'),


    # metadata to display on PyPI
    author="Vikas Dhiman",
    author_email="wecacuee@github.com",
    description=readlines("README.md"),
    license="MIT",
    keywords="mujoco mujoco_py derivative",
    url="https://github.com/wecacuee/mujoco_py_deriv",   # project home page, if any
)

