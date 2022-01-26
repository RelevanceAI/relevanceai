"""Relevance AI's base Client class - primarily used to login and access
the Dataset class or Clusterer class.


The recomended way to log in is using: 

.. code-block::

    from relevanceai import Client 
    client = Client()
    client.list_datasets()

If the user already knows their project and API key, they can 
log in this way: 

.. code-block::
    
    from relevanceai import Client 
    project = ""
    api_key = ""
    client = Client(project=project, api_key=api_key)
    client.list_datasets()

"""
import getpass
import json
import os
from typing import Union, Optional

from doc_utils.doc_utils import DocUtils
from relevanceai.dataset_api.dataset import Dataset, Datasets
from relevanceai.clusterer import Clusterer, KMeansClusterer, ClusterBase

from relevanceai.errors import APIError
from relevanceai.api.client import BatchAPIClient
from relevanceai.config import CONFIG
from relevanceai.vector_tools.plot_text_theme_model import build_and_plot_clusters


vis_requirements = False
try:
    from relevanceai.visualise.projector import Projector

    vis_requirements = True

except ModuleNotFoundError as e:
    # warnings.warn(f"{e} You can fix this by installing RelevanceAI[vis]")
    pass

from relevanceai.vector_tools.client import VectorTools


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1")


class Client(BatchAPIClient, DocUtils):
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

        # Authenticate user
        if authenticate:
            if self.check_auth():

                WELCOME_MESSAGE = f"""Welcome to the RelevanceAI Python SDK. Logged in as {project}."""
                print(WELCOME_MESSAGE)
            else:
                raise APIError(self.FAIL_MESSAGE)

        # Import projector and vector tools
        if vis_requirements:
            self.projector = Projector(project, api_key)
        else:
            self.logger.warning(
                "Projector not loaded. You do not have visualisation requirements installed."
            )
        self.vector_tools = VectorTools(project, api_key)

        self.Dataset = Dataset(project=project, api_key=api_key)
        self.Datasets = Datasets(project=project, api_key=api_key)

    # @property
    # def output_format(self):
    #     return CONFIG.get_field("api.output_format", CONFIG.config)

    # @output_format.setter
    # def output_format(self, value):
    #     CONFIG.set_option("api.output_format", value)

    ### Configurations

    @property
    def base_url(self):
        return CONFIG.get_field("api.base_url", CONFIG.config)

    @base_url.setter
    def base_url(self, value):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_url", value)

    @property
    def base_ingest_url(self):
        return CONFIG.get_field("api.base_ingest_url", CONFIG.config)

    @base_ingest_url.setter
    def base_ingest_url(self, value):
        if value.endswith("/"):
            value = value[:-1]
        CONFIG.set_option("api.base_ingest_url", value)

    ### Authentication Details

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
        project, api_key = self._token_to_auth()
        return Client(project=project, api_key=api_key, authenticate=authenticate)

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_search_suggestion(self):
        return self.services.search.make_suggestion()

    def check_auth(self):
        return self.admin._ping()

    ### Utility functions

    build_and_plot_clusters = build_and_plot_clusters

    ### CRUD-related utility functions

    def list_datasets(self):
        """List Datasets

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.list_datasets()

        """
        return self.datasets.list()

    def delete_dataset(self, dataset_id):
        """
        Delete a dataset

        Parameters
        ------------
        dataset_id: str
            The ID of a dataset

        Example
        ---------

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.delete_dataset("sample_dataset")

        """
        return self.datasets.delete(dataset_id)

    ### Clustering

    def Clusterer(
        self,
        model: ClusterBase,
        alias: str,
        cluster_field: str = "_cluster_",
    ):
        return Clusterer(
            model=model,
            alias=alias,
            cluster_field=cluster_field,
            project=self.project,
            api_key=self.api_key,
        )

    def KMeansClusterer(
        self,
        alias: str,
        k: Union[None, int] = 10,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = True,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str = "auto",
        cluster_field: str = "_cluster_",
    ):
        return KMeansClusterer(
            alias=alias,
            k=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm,
            cluster_field=cluster_field,
            project=self.project,
            api_key=self.api_key,
        )
