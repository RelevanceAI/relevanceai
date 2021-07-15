"""API Client
"""
from .datasets import Datasets
from .services import Services
from ..config import Config, CONFIG

class APIClient:
    """API Client
    """
    config: Config = CONFIG
    def __init__(self, project: str, api_key: str, base_url: str):
        self.datasets = Datasets(
                project=project, api_key=api_key, base_url=base_url)
        self.services = Services(
                project=project, api_key=api_key, base_url=base_url)
