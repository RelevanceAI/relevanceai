import os
from setuptools import setup, find_packages

def read(rel_path):
    """Read lines from given file"""
    here = os.path.abspath(os.path.dirname(__file__))
    with open(os.path.join(here, rel_path), "r") as fp:
        return fp.read()

def get_version(rel_path):
    """Read __version__ from given file"""
    for line in read(rel_path).splitlines():
        if line.startswith("__version__"):
            delim = '"' if '"' in line else "'"
            return line.split(delim)[1]
    raise RuntimeError(f"Unable to find a valid __version__ string in {rel_path}.")

setup(
    name='VecDB',
    version=get_version("vecdb/_version.py"),
    url='',
    author='Relevance AI',
    author_email='dev@vctr.ai',
    description='No description',
    packages=find_packages(),    
    install_requires=[],
)

