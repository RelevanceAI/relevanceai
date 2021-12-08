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
from relevanceai.errors import APIError, ClusteringResultsAlreadyExistsError
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
    EXISTING_CLUSTER_MESSAGE = """Clustering results already exist"""

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

    def kmeans_cluster(self,
        dataset_id: str,
        vector_fields: list,
        filters: List = [],
        k: Union[None, int] = 10,
        init: str = "k-means++",
        n_init: int = 10,
        max_iter: int = 300,
        tol: float = 1e-4,
        verbose: bool = True,
        random_state: Optional[int] = None,
        copy_x: bool = True,
        algorithm: str ="auto",
        alias: str = "default",
        cluster_field: str="_cluster_",
        update_documents_chunksize: int = 50,
        overwrite: bool = False
    ):
        """
        This function performs all the steps required for Kmeans clustering:
        1- Loads the data
        2- Clusters the data
        3- Updates the data with clustering info
        4- Adds the centroid to the hidden centroid collection

        Parameters
        ----------
        dataset_id : string
            name of the dataser
        vector_fields : list
            a list containing the vector field to be used for clustering
        filters : list
            a list to filter documents of the dataset,
        k : int
            K in Kmeans
        init : string
            "k-means++" -> Kmeans algorithm parameter
        n_init : int
            number of reinitialization for the kmeans algorithm
        max_iter : int
            max iteration in the kmeans algorithm
        tol : int
            tol in the kmeans algorithm
        verbose : bool
            True by default
        random_state = None
            None by default -> Kmeans algorithm parameter
        copy_x : bool
            True bydefault
        algorithm : string
            "auto" by default
        alias : string
            "default", string to be used in naming of the field showing the clustering results
        cluster_field: string
            "_cluster_", string to name the main cluster field
        overwrite : bool
            False by default, To overwite an existing clusering result

        """
        if '.'.join([cluster_field, vector_fields[0], alias]) in self.datasets.schema(dataset_id) and not overwrite:
            raise ClusteringResultsAlreadyExistsError(self.EXISTING_CLUSTER_MESSAGE)

        # load the documents
        docs = self.get_all_documents(dataset_id=dataset_id, filters=filters, select_fields=vector_fields)

        # Cluster
        clusterer = KMeans(
            k=k,
            init=init,
            n_init=n_init,
            max_iter=max_iter,
            tol=tol,
            verbose=verbose,
            random_state=random_state,
            copy_x=copy_x,
            algorithm=algorithm
        )
        clustered_docs = clusterer.fit_documents(
            vector_fields,
            docs,
            alias=alias, 
            cluster_field=cluster_field, 
            return_only_clusters=True)

        # Updating the db
        try:
            results = self.update_documents(dataset_id, clustered_docs, chunksize = update_documents_chunksize)
        except Exception as e:
            self.logger.error(e)
        self.logger.info(results)

        # Update the centroid collection
        centers = clusterer.get_centroid_docs()
        try:
            results = self.services.cluster.centroids.insert(
                dataset_id = dataset_id,
                cluster_centers=centers,
                vector_field=vector_fields[0],
                alias= alias
            )
        except Exception as e:
            self.logger.error(e)
        self.logger.info(results)
        return centers
