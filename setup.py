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


requirements = [
    "tqdm>=4.49.0",
    "pandas>=1.0.0",
    "loguru>=0.5.3",
    "document-utils>=1.3.0",
    "requests>=2.0.0",
    "typing-extensions"
]

excel_requirements = [
    "openpyxl>=3.0.9", 
    "fsspec>=2021.10.1"
]

vis_requirements = [
    "scikit-learn", 
    # "umap-learn>=0.5.2",
    "ivis[cpu]>=2.0.6",
    "plotly>=5.3.1",
    "kmodes>=0.11.1"
]

test_requirements =[
    "pytest",
    "pytest-dotenv",
    "pytest-cov",
] + excel_requirements \
  + vis_requirements

dev_requirements = [
    "autopep8",
    "pylint",
] + test_requirements

setup(
    name="VecDB",
    version=get_version("vecdb/__init__.py"),
    url="",
    
    author="Relevance AI",
    author_email="dev@vctr.ai",
    long_description="",

    package_dir={"": "vecdb"},
    packages=find_packages(where="vecdb"),

    setup_requires=["wheel"],
    install_requires=requirements,
    package_data={'': ['*.ini',]},
    extras_require={
        "dev": dev_requirements,
        "excel": excel_requirements,
        "vis": vis_requirements,
        "tests": test_requirements,
        "notebook": ["jsonshower"],
    },
    python_requires=">=3.6",
    classifiers=[],
)
