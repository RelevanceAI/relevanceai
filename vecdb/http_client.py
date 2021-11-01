"""access the client via this class
"""
import os
import sys

from doc_utils import DocUtils
from loguru import logger

from .batch.client import BatchAPIClient
from .config import CONFIG
from .errors import APIError


class VecDBClient(BatchAPIClient, DocUtils):
    """Python Client for Relevance AI's VecDB"""

    WELCOME_MESSAGE = """Welcome to the development version of the VecDB Python SDK. Please do not use this SDK to build core systems yet until we have a stable version."""
    FAIL_MESSAGE = """Your Api-Key is invalid. Please login again"""
    config = CONFIG

    def __init__(
        self,
        project: str = os.getenv("VDB_PROJECT"),
        api_key: str = os.getenv("VDB_API_KEY"),
        base_url: str = "https://gateway-api-aueast.relevance.ai/v1/",
    ):
        super().__init__(project, api_key, base_url)

        # Add Logging
        self.logger = logger
        self.logger.remove()
        self.logger.add(sys.stdout, level=CONFIG.logging_level)
        if CONFIG.log_to_file:
            logger.add(
                "vecdb_{time}.log", level=CONFIG.logging_level, rotation="100 MB"
            )

        if project is None or api_key is None:
            print(
                "It seems you are missing an API key, "
                + "you can sign up for an API key following the instructions here: "
                + "https://discovery.relevance.ai/reference/usage"
            )

        if (
            self.datasets.list(
                verbose=False, output_format=False, retries=1
            ).status_code
            == 200
        ):
            self.logger.info(self.WELCOME_MESSAGE)
        else:
            raise APIError(self.FAIL_MESSAGE)

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}
