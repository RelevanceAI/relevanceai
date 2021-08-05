"""access the client via this class
"""
import os
from .config import CONFIG
from .batch.client import BatchAPIClient

class VecDBClient(BatchAPIClient):
    """VecDB Client
    """
    WELCOME_MESSAGE = """Welcome to the development version of the VecDB Python SDK. Please do not use this SDK to build core systems yet until we have a stable version."""
    config = CONFIG
    def __init__(
        self,
        project: str=os.getenv("VDB_PROJECT"), 
        api_key: str=os.getenv("VDB_API_KEY"),
        base_url: str="https://api-dev-aueast.relevance.ai/v1/"):
        print(self.WELCOME_MESSAGE)
        super().__init__(project, api_key, base_url)
    
    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}
