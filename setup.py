import numpy as np
from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext

ext_modules=[
    Extension("hausdorff",
              ["hausdorff.pyx"],
              libraries=["m"],
              extra_compile_args = ["-O3", "-ffast-math", "-march=native"],
              extra_link_args=[]
              ) 
]

cythonize("hausdorff.pyx")

setup( 
  name = "hausdorff",
  include_dirs = [np.get_include(),'include'],
  cmdclass = {"build_ext": build_ext},
  ext_modules = ext_modules
)