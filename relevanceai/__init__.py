# -*- coding: utf-8 -*-
import warnings
from relevanceai.client import Client

# Cluster _Base Utilities
from relevanceai.operations.cluster.base import (
    ClusterBase,
    CentroidClusterBase,
)
from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.dr.ops import ReduceDimensionsOps
from relevanceai.operations.vector import Base2Vec

# Fix the name
from relevanceai.utils import datasets
from relevanceai.utils.datasets import mock_documents
import requests

from relevanceai.constants.warning import Warning

# Import useful utility if possible as well
try:
    from jsonshower import show_json
except ModuleNotFoundError:
    pass

__version__ = "3.2.4"

try:
    pypi_data = requests.get("https://pypi.org/pypi/relevanceai/json").json()
    pypi_info = pypi_data.get("info", None)
    if pypi_info is not None:
        latest_version = pypi_info.get("version", None)
    else:
        latest_version = None

    if __version__ < latest_version and latest_version is not None:
        changelog_url: str = (
            f"https://relevanceai.readthedocs.io/en/{__version__}/changelog.html"
        )
        warnings.warn(
            Warning.LATEST_VERSION.format(
                version=__version__,
                latest_version=latest_version,
                changelog_url=changelog_url,
            )
        )
except:
    pass
