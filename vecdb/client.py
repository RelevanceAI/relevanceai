"""access the client via this class
"""
import os
import requests
import time
from .helpers import HelperMixin
from .datasets import Datasets
from .services import Services
from .config import Config

class VecDBClient(HelperMixin):
    """VecDB Client
    """
    def __init__(
        self,
        project: str=os.getenv("VDB_PROJECT"), 
        api_key: str=os.getenv("VI_API_KEY"),
        base_url: str="https://api-dev-aueast.relevance.ai/v1/",
        local: bool=False):
        """
        Params:
            local: if local is True, then the base_url switches to the default locally hosted one.
        """
        self.project = project
        self.api_key = api_key
        if local: 
            self.base_url = "http://localhost:8000/v1/"
        else:
            self.base_url = base_url
        self.config = Config()
        self.datasets = Datasets(project=project, api_key=api_key, base_url=base_url)
        self.services = Services(project=project, api_key=api_key, base_url=base_url)
