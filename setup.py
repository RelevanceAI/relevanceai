import os

from setuptools import find_packages, setup


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

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name='VecDB',
    version=get_version("vecdb/__init__.py"),
    url='',
    author='Relevance AI',
    author_email='dev@vctr.ai',
    description='No description',
    packages=find_packages(),    
    install_requires=required,
    extras_require={
        "tests": ["pytest"]
    }
)

