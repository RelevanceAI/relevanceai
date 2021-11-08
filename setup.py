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


requirements = [
    "tqdm>=4.49.0",
    "pandas>=1.0.0",
    "loguru>=0.5.3",
    "document-utils>=1.3.0",
    "requests>=2.0.0",
    "numpy>=1.19.0",
]

dev_requirements = [
    "autopep8",
    "pylint",
    "pytest",
    "pytest-dotenv",
    "pytest-cov",
    "pytest-mock",
    "sphinx-rtd-theme>=0.5.0"
]

setup(
    name="RelevanceAI",
    version=get_version("relevanceai/__init__.py"),
    url="",
    author="Relevance AI",
    author_email="dev@relevance.ai",
    long_description="",
    packages=find_packages(),
    install_requires=requirements,
    package_data={
        "": [
            "*.ini",
        ]
    },
    extras_require={
        "dev": dev_requirements,
        "excel": ["fsspec==2021.10.1", "openpyxl==3.0.9"],
        "tests": ["pytest", "fsspec==2021.10.1", "openpyxl==3.0.9"],
        "notebook": ["jsonshower"],
    },
    # python_requires=">=3.7",
    classifiers=[],
)
