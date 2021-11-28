"""access the client via this class
"""
import getpass
import os
from typing import Optional

from doc_utils.doc_utils import DocUtils

from relevanceai.batch.client import BatchAPIClient
from relevanceai.config import CONFIG
from relevanceai.errors import APIError

vis_requirements = False
try:
    from relevanceai.visualise.projector import Projector

    vis_requirements = True
except ModuleNotFoundError as e:
    pass


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Client(BatchAPIClient, DocUtils):
    """Python Client for Relevance AI's relevanceai"""

    WELCOME_MESSAGE = """Welcome to the RelevanceAI Python SDK"""
    FAIL_MESSAGE = """Your API key is invalid. Please login again"""

    def __init__(
        self,
        project: Optional[str]=os.getenv("VDB_PROJECT", None),
        api_key: Optional[str]=os.getenv("VDB_API_KEY", None),
        base_url: Optional[str]="https://gateway-api-aueast.relevance.ai/v1",
        verbose: bool=True

    ):

        if project is None or api_key is None:
            project, api_key = Client.token_to_auth(verbose=verbose)
            # raise ValueError(
            #     "It seems you are missing an API key, "
            #     + "you can sign up for an API key following the instructions here: "
            #     + "https://discovery.relevance.ai/reference/usage"
            # )

        # if (
        #     self.datasets.list(
        #     ).status_code
        #     == 200
        # ):
        #     if verbose: self.logger.success(self.WELCOME_MESSAGE)
        # else:
        # raise APIError(self.FAIL_MESSAGE)
        if verbose:
            self.logger.success(self.WELCOME_MESSAGE)

        super().__init__(project, api_key, base_url)  # type: ignore
        if vis_requirements:
            self.projector = Projector(project, api_key, base_url)

    @staticmethod
    def token_to_auth(verbose=True):
        if verbose:
            print("You can sign up/login and find your credentials here: https://auth.relevance.ai/signup/?callback=https%3A%2F%2Fcloud.relevance.ai%2Flogin%3Fredirect%3Dcli-api")
            print("Once you have signed up, click on the value under `Authorization token` and paste it here:")
        token = getpass.getpass("Authorization token (you can find it here: https://auth.relevance.ai/signup/?callback=https%3A%2F%2Fcloud.relevance.ai%2Flogin%3Fredirect%3Dcli-api")
        # project = getpass.getpass("Project:")
        # api_key = getpass.getpass("API key:")
        # token = getpass.getpass(
        #     "Paste your project and API key in the format: of `project:api_key` here:"
        # )
        project = token.split(":")[0]
        api_key = token.split(":")[1]
        return project, api_key

    @staticmethod
    def login(
        base_url: str = "https://gateway-api-aueast.relevance.ai/v1",
        verbose: bool = True,
    ):
        """Preferred login method for demos and interactive usage."""
        project, api_key = Client.token_to_auth()
        return Client(
            project=project, api_key=api_key, base_url=base_url, verbose=verbose
        )

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_search_suggestion(self):
        return self.services.search.make_suggestion() 

