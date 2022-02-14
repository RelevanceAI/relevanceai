# -*- coding: utf-8 -*-
import warnings
from relevanceai import vector_tools
from relevanceai.http_client import Client

# Cluster _Base Utilities
from relevanceai.clusterer.cluster_base import ClusterBase, CentroidClusterBase
from relevanceai.clusterer.clusterer import ClusterOps
import requests

# Import useful utility if possible as well
try:
    from jsonshower import show_json
except ModuleNotFoundError:
    pass

__version__ = "1.1.1"

try:
    pypi_data = requests.get("https://pypi.org/pypi/relevanceai/json").json()
    pypi_info = pypi_data.get("info", None)
    if pypi_info is not None:
        latest_version = pypi_info.get("version", None)
    else:
        latest_version = None

    if __version__ != latest_version and latest_version is not None:
        changelog_url: str = (
            f"https://relevanceai.readthedocs.io/en/{__version__}/changelog.html"
        )
        MESSAGE = """We noticed you don't have the latest version! 
We recommend updating to the latest version ({latest_version}) to get all bug fixes and newest features!
You can do this by running pip install -U relevanceai. 
Changelog: {changelog_url}.""".format(  # type: ignore
            version=__version__,
            latest_version=latest_version,
            changelog_url=changelog_url,
        )
        warnings.warn(MESSAGE)
except:
    pass
