"""access the client via this class
"""
import getpass
import os
import sys
from typing import Optional, List, Union

from doc_utils.doc_utils import DocUtils

from relevanceai.api.client import BatchAPIClient
from relevanceai.api.endpoints.cluster import Cluster
from relevanceai.config import CONFIG
from relevanceai.vector_tools.cluster import KMeans

vis_requirements = False
try:
    from relevanceai.visualise.projector import Projector

    vis_requirements = True
except ModuleNotFoundError as e:
    pass

from relevanceai.vector_tools.client import VectorTools

def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Client(BatchAPIClient, DocUtils):
    """Python Client for Relevance AI's relevanceai"""

    WELCOME_MESSAGE = """Welcome to the RelevanceAI Python SDK"""
    FAIL_MESSAGE = """Your API key is invalid. Please login again"""

    def __init__(
        self,
        project = os.getenv("RELEVANCE_PROJECT"),
        api_key = os.getenv("RELEVANCE_API_KEY"),
        verbose: bool=True
    ):

        if project is None or api_key is None:
            project, api_key = Client.token_to_auth(verbose=verbose)

        super().__init__(project, api_key)

        # if self.check_auth():
        #     if verbose: self.logger.success(self.WELCOME_MESSAGE)
        # else:
        #     raise APIError(self.FAIL_MESSAGE)

        if vis_requirements:
            self.projector = Projector(project, api_key)

        self.vector_tools = VectorTools(project, api_key)

    @property
    def output_format(self):
        return CONFIG.get_field("api.output_format", CONFIG.config)

    @output_format.setter
    def output_format(self, value):
        CONFIG.set_option("api.output_format", value)

    @staticmethod
    def token_to_auth(verbose=True):
        # if verbose:
        #     print("You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api")
        #     print("Once you have signed up, click on the value under `Authorization token` and paste it here:")
        # SIGNUP_URL = "https://auth.relevance.ai/signup/?callback=https%3A%2F%2Fcloud.relevance.ai%2Flogin%3Fredirect%3Dcli-api"
        SIGNUP_URL = "https://cloud.relevance.ai/sdk/api"
        token = getpass.getpass(f"Authorization token (you can find it here: {SIGNUP_URL})")
        project = token.split(":")[0]
        api_key = token.split(":")[1]
        os.environ["RELEVANCE_PROJECT"] = project
        os.environ["RELEVANCE_API_KEY"] = api_key
        return project, api_key

    @staticmethod
    def login(
        verbose: bool = True,
    ):
        """Preferred login method for demos and interactive usage."""
        project, api_key = Client.token_to_auth()
        return Client(
            project=project, api_key=api_key, verbose=verbose
        )

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_search_suggestion(self):
        return self.services.search.make_suggestion() 

    def check_auth(self):
        """TODO: Add a proper way to check authentication based on pinging.
        """
        response = self.datasets.list()
        try:
            return response.status_code == 200    
        except:
            raise Exception("Invalid auth details.")
