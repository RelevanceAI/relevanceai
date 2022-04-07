import os
from datetime import datetime
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
    "document-utils>=1.7.1",
    "requests>=2.0.0",
    "numpy>=1.19.0",
    "joblib>=1.0.0",
    "scikit-learn>=0.20.0",  # last version of support to Python3.4
    "typing-extensions>=3.0",
    "analytics-python~=1.4.0",
    "aiohttp>=3.8.1",
    "appdirs>=1.4.4",
    "orjson>=3.6.7",
]

excel_requirements = requirements + ["openpyxl>=3.0.9", "fsspec>=2021.10.1"]

vis_requirements = requirements + [
    # "matplotlib>=3.5.1",
    # "plotly>=5.5.0",
    # "typeguard",
    # "dash",
    # "pillow",
    # "opencv-python",
    # "scikit-image",
    # "dash_bootstrap_components",
]

umap = ["umap-learn>=0.5.2"]
# ivis_cpu = ["ivis[cpu]>=2.0.6"]
# ivis_gpu = ["ivis[gpu]>=2.0.6"]
kmedoids = ["scikit-learn-extra>=0.2.0"]
hdbscan = ["hdbscan>=0.8.27"]

# vis_extras = umap + ivis_cpu + ivis_gpu + kmedoids + hdbscan

test_requirements = (
    [
        "pytest",
        "pytest-dotenv",
        "pytest-xdist",
        "pytest-cov",
        "pytest-mock",
        "mypy",
        "types-requests",
        "pytest-sugar",
        "pytest-rerunfailures",
    ]
    + excel_requirements
    + vis_requirements
    + requirements
    # + vis_extras
)

doc_requirements = [
    "sphinx-rtd-theme>=0.5.0",
    "pydata-sphinx-theme==0.8.1",
    "sphinx-autoapi==1.8.4",
    "sphinx-autodoc-typehints==1.12.0",
]

dev_requirements = (
    ["autopep8", "pylint", "jupyter", "pre-commit", "black", "mypy"]
    + test_requirements
    + doc_requirements
)


dev_vis_requirements = (
    ["autopep8", "pylint", "jupyter"]
    + test_requirements
    + vis_requirements
    + doc_requirements
    # + vis_extras
)

from pathlib import Path

this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

name = "RelevanceAI"
version = get_version("relevanceai/__init__.py")

if os.getenv("_IS_DEV"):
    name = "RelevanceAI-dev"
    version = (
        version
        + "."
        + datetime.now().__str__().replace("-", ".").replace(" ", ".").replace(":", ".")
    )

setup(
    name=name,
    version=version,
    url="https://relevance.ai/",
    author="Relevance AI",
    author_email="dev@relevance.ai",
    long_description=long_description,
    long_description_content_type="text/markdown",
    packages=find_packages(),
    setup_requires=["wheel"],
    install_requires=requirements,
    package_data={
        "": [
            "*.ini",
        ]
    },
    extras_require={
        "docs": doc_requirements,
        "dev": dev_requirements,
        "dev-vis": dev_vis_requirements,
        "dev-viz": dev_vis_requirements,
        "excel": excel_requirements,
        "vis": vis_requirements,
        "viz": vis_requirements,
        # "vis-all": vis_requirements + vis_extras,
        "tests": test_requirements,
        "notebook": ["jsonshower"] + vis_requirements,
        "umap": umap,
        # "ivis-cpu": ivis_cpu,
        # "ivis-gpu": ivis_gpu,
        "kmedoids": kmedoids,
        "hdbscan": hdbscan,
    },
    python_requires=">=3.6",
    classifiers=[
        "Development Status :: 5 - Production/Stable",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Information Technology",
        "Intended Audience :: Financial and Insurance Industry",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Manufacturing",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: OS Independent",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: Implementation :: PyPy",
        "Topic :: Database",
        "Topic :: Internet :: WWW/HTTP :: Indexing/Search",
        "Topic :: Multimedia :: Sound/Audio :: Conversion",
        "Topic :: Multimedia :: Video :: Conversion",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Image Recognition",
        "Topic :: Scientific/Engineering :: Information Analysis",
        "Topic :: Scientific/Engineering :: Visualization",
        "Topic :: Software Development :: Libraries :: Application Frameworks",
    ],
)
