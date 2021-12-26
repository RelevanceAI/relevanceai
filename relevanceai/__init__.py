from relevanceai import vector_tools
from relevanceai.http_client import Client
import requests

# Import useful utility if possible as well
try:
    from jsonshower import show_json
except ModuleNotFoundError:
    pass

__version__ = "0.24.8"

latest_version = requests.get("https://pypi.org/pypi/relevanceai/json").json()["info"][
    "version"
]
if __version__ != latest_version:
    print(
        f"Your RelevanceAI version ({__version__}) is not the latest. Please install the latest version ({latest_version}) by running pip install -U relevanceai"
    )
