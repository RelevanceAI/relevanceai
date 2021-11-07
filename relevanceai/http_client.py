"""access the client via this class
"""
import os

from doc_utils import DocUtils

from .batch.client import BatchAPIClient
from .errors import APIError


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Client(BatchAPIClient, DocUtils):
    """Python Client for Relevance AI's relevanceai"""

    WELCOME_MESSAGE = """Welcome to the development version of the relevanceai Python SDK"""
    FAIL_MESSAGE = """Your API key is invalid. Please login again"""

    def __init__(
        self,
        project: str = os.getenv("VDB_PROJECT"),
        api_key: str = os.getenv("VDB_API_KEY"),
        base_url: str = "https://gateway-api-aueast.relevance.ai/v1/",
    ):
        super().__init__(project, api_key, base_url)

        if project is None or api_key is None:
            raise ValueError(
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
            self.logger.success(self.WELCOME_MESSAGE)
        else:
            raise APIError(self.FAIL_MESSAGE)

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}
