"""access the client via this class
"""
import getpass
import json
import os
import sys
import warnings
from typing import Optional, List, Union

from doc_utils.doc_utils import DocUtils

from relevanceai.errors import APIError
from relevanceai.api.client import BatchAPIClient
from relevanceai.api.endpoints.cluster import ClusterClient
from relevanceai.config import CONFIG
from relevanceai.vector_tools.cluster import KMeans

vis_requirements = False
try:
    from relevanceai.visualise.projector import Projector

    vis_requirements = True
except ModuleNotFoundError as e:
    warnings.warn(f"{e} You can fix this by installing RelevanceAI[vis]")
    pass

from relevanceai.vector_tools.client import VectorTools


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Client(BatchAPIClient, DocUtils):
    """Python Client for Relevance AI's relevanceai"""

    FAIL_MESSAGE = """Your API key is invalid. Please login again"""
    _cred_fn = ".creds.json"

    def __init__(
        self,
        project=os.getenv("RELEVANCE_PROJECT"),
        api_key=os.getenv("RELEVANCE_API_KEY"),
        authenticate: bool = False,
    ):

        if project is None or api_key is None:
            project, api_key = self._token_to_auth()

        super().__init__(project, api_key)

        if authenticate:
            if self.check_auth():

                WELCOME_MESSAGE = f"""Welcome to the RelevanceAI Python SDK. Logged in as {project}."""
                print(self.WELCOME_MESSAGE)
            else:
                raise APIError(self.FAIL_MESSAGE)

        if vis_requirements:
            self.projector = Projector(project, api_key)
        else:
            self.logger.warning(
                "Projector not loaded. You do not have visualisation requirements installed."
            )
        self.vector_tools = VectorTools(project, api_key)

    # @property
    # def output_format(self):
    #     return CONFIG.get_field("api.output_format", CONFIG.config)

    # @output_format.setter
    # def output_format(self, value):
    #     CONFIG.set_option("api.output_format", value)

    @property
    def base_url(self):
        return CONFIG.get_field("api.base_url", CONFIG.config)

    @base_url.setter
    def base_url(self, value):
        CONFIG.set_option("api.base_url", value)

    def _token_to_auth(self):
        # if verbose:
        #     print("You can sign up/login and find your credentials here: https://cloud.relevance.ai/sdk/api")
        #     print("Once you have signed up, click on the value under `Authorization token` and paste it here:")
        # SIGNUP_URL = "https://auth.relevance.ai/signup/?callback=https%3A%2F%2Fcloud.relevance.ai%2Flogin%3Fredirect%3Dcli-api"
        SIGNUP_URL = "https://cloud.relevance.ai/sdk/api"
        if not os.path.exists(self._cred_fn):
            # We repeat it twice because of different behaviours
            print(f"Authorization token (you can find it here: {SIGNUP_URL} )")
            token = getpass.getpass(f"Auth token:")
            project = token.split(":")[0]
            api_key = token.split(":")[1]
            self._write_credentials(project, api_key)
        else:
            data = self._read_credentials()
            project = data["project"]
            api_key = data["api_key"]
        return project, api_key

    def _write_credentials(self, project, api_key):
        json.dump({"project": project, "api_key": api_key}, open(self._cred_fn, "w"))

    def _read_credentials(self):
        return json.load(open(self._cred_fn))

    def login(
        self,
        authenticate: bool = True,
    ):
        """Preferred login method for demos and interactive usage."""
        project, api_key = self._token_to_auth()
        return Client(project=project, api_key=api_key, authenticate=authenticate)

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_search_suggestion(self):
        return self.services.search.make_suggestion()

    def check_auth(self):
        return self.admin._ping()
