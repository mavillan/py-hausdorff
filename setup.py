from distutils.core import setup
from setuptools import find_packages

setup( 
  name = "hausdorff",
  version = '0.2.1',
  author = 'mavillan',
  author_email = 'nallivam@gmail.com',
  packages=find_packages(),
  license='GNU GENERAL PUBLIC LICENSE Version 3',
  long_description="""
  This package implements the algorithm presented in An Efficient Algorithm for Calculating the Exact Hausdorff Distance (DOI: [10.1109/TPAMI.2015.2408351](https://doi.org/10.1109/TPAMI.2015.2408351)) by Aziz and Hanbury, 2015.
  """,
)
