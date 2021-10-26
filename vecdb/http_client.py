"""access the client via this class
"""
import os
from loguru import logger
from .config import CONFIG
from .batch.client import BatchAPIClient
from .errors import APIError
from doc_utils import DocUtils

class VecDBClient(BatchAPIClient, DocUtils):
    """VecDB Client
    """
    WELCOME_MESSAGE = """Welcome to the development version of the VecDB Python SDK. Please do not use this SDK to build core systems yet until we have a stable version."""
    FAIL_MESSAGE = """Your Api-Key is invalid. Please login again"""
    config = CONFIG
    def __init__(
        self,
        project: str=os.getenv("VDB_PROJECT"), 
        api_key: str=os.getenv("VDB_API_KEY"),
        base_url: str="https://gateway-api-aueast.relevance.ai/v1/"):
        super().__init__(project, api_key, base_url)
        self.logger = logger
        if project is None or api_key is None:
            print("It seems you are missing an API key, " + \
                    "you can sign up for an API key at " + \
                    "https://discovery.relevance.ai/reference/quick-start.")
        
        if (self.datasets.list(verbose=False, output_format = False, retries=1).status_code == 200):
            self.logger.info(self.WELCOME_MESSAGE)
        else:
            raise APIError(self.FAIL_MESSAGE)

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}
