from setuptools import setup, find_packages
from os.path import join, dirname

import pyrecsys

setup(
    name='pyrecsys',
    version=pyrecsys.__version__,
    packages=find_packages(),
    author = "Vladimir Larin",
    author_email = "vladimir@vlarine.ru",
    description = "Collaborative filtering recommender system",
    license = "MIT",
    url = "https://github.com/vlarine/pyrecsys",
    long_description=open(join(dirname(__file__), 'README.md')).read(),
)
