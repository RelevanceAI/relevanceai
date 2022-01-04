import warnings
from relevanceai import vector_tools
from relevanceai.http_client import Client
import requests

# Import useful utility if possible as well
try:
    from jsonshower import show_json
except ModuleNotFoundError:
    pass

__version__ = "0.24.9"

try:
    pypi_data = requests.get("https://pypi.org/pypi/relevanceai/json").json()
    pypi_info = pypi_data.get("info", None)
    if pypi_info is not None:
        latest_version = pypi_info.get("version", None)
    else:
        latest_version = None

    if __version__ != latest_version and latest_version is not None:
        warnings.warn(
            "Your RelevanceAI version ({version}) is not the latest. Please install the latest version ({latest_version}) by running pip install -U relevanceai".format(
                version=__version__, latest_version=latest_version
            )
        )
except:
    pass
