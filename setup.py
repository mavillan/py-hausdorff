from distutils.core import setup
from setuptools import find_packages

setup( 
  name = "hausdorff",
  version = '0.2',
  author = 'mavillan',
  author_email = 'nallivam@gmail.com',
  packages=find_packages(),
  license='GNU GENERAL PUBLIC LICENSE Version 3',
  long_description=open('./README.md').read(),
)
