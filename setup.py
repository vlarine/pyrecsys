from setuptools import setup, find_packages
from os.path import join, dirname

import pyrecsys

setup(
    name='pyrecsys',
    version=pyrecsys.__version__,
    packages=find_packages(),
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)
