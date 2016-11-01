from setuptools import setup, find_packages
from os.path import join, dirname

import pyrecsys

with open('requirements.txt') as f:
    reqs = f.read().splitlines()

setup(
    name='pyrecsys',
    version=pyrecsys.__version__,
    packages=['pyrecsys', 'pyrecsys._polara.lib'],
    author = "Vladimir Larin",
    author_email = "vladimir@vlarine.ru",
    description = "Collaborative filtering recommender system",
    license = "MIT",
    url = "https://github.com/vlarine/pyrecsys",
    long_description=open(join(dirname(__file__), 'README.md')).read(),
    install_requires=reqs,
)
