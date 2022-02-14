"""Relevance AI's base Client class - primarily used to login and access
the Dataset class or ClusterOps class.


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
    client = Client(project=project, api_key=api_key, firebase_uid=firebase_uid)
    client.list_datasets()

If you need to change your token, simply run:

.. code-block::

    from relevanceai import Client
    client = Client(token="...")

"""
import os
import getpass
from base64 import b64decode as decode
from typing import Optional, List, Dict, Union

from doc_utils.doc_utils import DocUtils
from relevanceai.dataset_api import Dataset
from relevanceai.clusterer import ClusterOps

from relevanceai.errors import APIError
from relevanceai.api.client import BatchAPIClient
from relevanceai.config import CONFIG
from relevanceai.vector_tools.plot_text_theme_model import build_and_plot_clusters
from functools import lru_cache

import analytics

from relevanceai.analytics_funcs import track, identify

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

    def __init__(
        self,
        project=os.getenv("RELEVANCE_PROJECT"),
        api_key=os.getenv("RELEVANCE_API_KEY"),
        firebase_uid=os.getenv("RELEVANCE_FIREBASE_UID"),
        region=None,
        authenticate: bool = True,
        token: str = None,
        force_refresh: bool = False,
    ):
        """
        Initialize the client

        Parameters
        -------------

        project: str
            The name of the project
        api_key: str
            API key
        region: str
            The region to work in. Currently only `us-east-1` is provided
        token: str
            You can paste the token here if things need to be refreshed
        force_refresh: bool
            If True, it forces you to refresh your client
        """

        try:
            self._set_mixpanel_write_key()
        except Exception as e:
            pass

        if project is None or api_key is None or force_refresh:
            credentials = self._token_to_auth(token)

        try:
            self.project = credentials["project"]
        except Exception:
            self.project = project

        try:
            self.api_key = credentials["api_key"]
        except Exception:
            self.api_key = api_key

        try:
            self.firebase_uid = credentials["firebase_uid"]
        except Exception:
            self.firebase_uid = firebase_uid

        self._identify()

        if region is not None:
            self.region = region

        self.base_url = self._region_to_url(self.region)
        self.base_ingest_url = self._region_to_ingestion_url(self.region)

        super().__init__(
            project=self.project, api_key=self.api_key, firebase_uid=self.firebase_uid
        )

        # used to debug
        if authenticate:
            if self.check_auth():
                WELCOME_MESSAGE = (
                    f"""Welcome to RelevanceAI. Logged in as {self.project}."""
                )
                print(WELCOME_MESSAGE)
            else:
                raise APIError(self.FAIL_MESSAGE)

        # Import projector and vector tools
        if vis_requirements:
            self.projector = Projector(
                project=self.project,
                api_key=self.api_key,
                firebase_uid=self.firebase_uid,
            )

        self.vector_tools = VectorTools(
            project=self.project, api_key=self.api_key, firebase_uid=self.firebase_uid
        )

        # Add non breaking changes to support old ways of inserting documents and csv
        self.insert_documents = Dataset(
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            dataset_id="",
        )._insert_documents
        self.insert_csv = Dataset(
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            dataset_id="",
        )._insert_csv

    # @property
    # def output_format(self):
    #     return CONFIG.get_field("api.output_format", CONFIG.config)

    # @output_format.setter
    # def output_format(self, value):
    #     CONFIG.set_option("api.output_format", value)

    ### Authentication Details

    @identify
    def _identify(self):
        return

    def _set_mixpanel_write_key(self):
        analytics.write_key = decode(self.mixpanel_write_key).decode("utf-8")

    def _process_token(self, token: str):
        split_token = token.split(":")
        project = split_token[0]
        api_key = split_token[1]
        if len(split_token) > 2:
            self.region = split_token[2]
            base_url = self._region_to_url(self.region)

            if len(split_token) > 3:
                firebase_uid = split_token[3]
                data = dict(
                    project=project,
                    api_key=api_key,
                    base_url=base_url,
                    firebase_uid=firebase_uid,
                )
                return data
            else:
                return dict(
                    project=project,
                    api_key=api_key,
                    base_url=base_url,
                )

        else:
            return dict(project=project, api_key=api_key)

    def _region_to_ingestion_url(self, region: str):
        # same as region to URL now in case ingestion ever needs to be separate
        if region == "old-australia-east":
            url = "https://gateway-api-aueast.relevance.ai/latest"
        else:
            url = f"https://api.{region}.relevance.ai/latest"
        return url

    def _token_to_auth(self, token: Optional[str] = None):
        SIGNUP_URL = "https://cloud.relevance.ai/sdk/api"
        if token:
            return self._process_token(token)
        else:
            print(f"Activation token (you can find it here: {SIGNUP_URL} )")
            if not token:
                token = self._get_token()
            return self._process_token(token)  # type: ignore

    def _get_token(self):
        # TODO: either use cache or keyring package
        token = getpass.getpass(f"Activation token:")
        return token

    @property
    def auth_header(self):
        return {"Authorization": self.project + ":" + self.api_key}

    def make_search_suggestion(self):
        return self.services.search.make_suggestion()

    def check_auth(self):
        print(f"Connecting to {self.region}...")
        return self.admin._ping()

    ### Utility functions

    build_and_plot_clusters = build_and_plot_clusters

    ### CRUD-related utility functions

    @track
    def create_dataset(self, dataset_id: str, schema: dict = {}):
        """
        A dataset can store documents to be searched, retrieved, filtered and aggregated (similar to Collections in MongoDB, Tables in SQL, Indexes in ElasticSearch).
        A powerful and core feature of VecDB is that you can store both your metadata and vectors in the same document. When specifying the schema of a dataset and inserting your own vector use the suffix (ends with) "_vector_" for the field name, and specify the length of the vector in dataset_schema. \n

        For example:

        .. code-block::
            {
                "product_image_vector_": 1024,
                "product_text_description_vector_" : 128
            }

        These are the field types supported in our datasets: ["text", "numeric", "date", "dict", "chunks", "vector", "chunkvector"]. \n

        For example:

        .. code-block::

            {
                "product_text_description" : "text",
                "price" : "numeric",
                "created_date" : "date",
                "product_texts_chunk_": "chunks",
                "product_text_chunkvector_" : 1024
            }

        You don't have to specify the schema of every single field when creating a dataset, as VecDB will automatically detect the appropriate data type for each field (vectors will be automatically identified by its "_vector_" suffix). Infact you also don't always have to use this endpoint to create a dataset as /datasets/bulk_insert will infer and create the dataset and schema as you insert new documents. \n

        Note:

            - A dataset name/id can only contain undercase letters, dash, underscore and numbers.
            - "_id" is reserved as the key and id of a document.
            - Once a schema is set for a dataset it cannot be altered. If it has to be altered, utlise the copy dataset endpoint.

        For more information about vectors check out the 'Vectorizing' section, services.search.vector or out blog at https://relevance.ai/blog. For more information about chunks and chunk vectors check out services.search.chunk.

        Parameters
        ----------
        dataset_id: str
            The unique name of your dataset
        schema : dict
            Schema for specifying the field that are vectors and its length

        Example
        ----------
        .. code-block::

            from relevanceai import Client
            client = Client()
            client.create_dataset("sample_dataset_id")

        """
        return self.datasets.create(dataset_id, schema=schema)

    @track
    def list_datasets(self):
        """List Datasets

        Example
        ----------

        .. code-block::

            from relevanceai import Client
            client = Client()
            client.list_datasets()

        """
        self.print_dashboard_message(
            "You can view all your datasets at https://cloud.relevance.ai/datasets."
        )
        return self.datasets.list()

    @track
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
            client.delete_dataset("sample_dataset_id")

        """
        return self.datasets.delete(dataset_id)

    def Dataset(
        self,
        dataset_id: str,
        fields: list = [],
        image_fields: List[str] = [],
        audio_fields: List[str] = [],
        highlight_fields: Dict[str, List] = {},
        text_fields: List[str] = [],
    ):
        return Dataset(
            dataset_id=dataset_id,
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
            fields=fields,
            image_fields=image_fields,
            audio_fields=audio_fields,
            highlight_fields=highlight_fields,
            text_fields=text_fields,
        )

    ### Clustering

    @track
    def ClusterOps(
        self,
        alias: str,
        model=None,
        dataset_id: Optional[str] = None,
        vector_fields: Optional[List[str]] = None,
        cluster_field: str = "_cluster_",
    ):
        return ClusterOps(
            model=model,
            alias=alias,
            dataset_id=dataset_id,
            vector_fields=vector_fields,
            cluster_field=cluster_field,
            project=self.project,
            api_key=self.api_key,
            firebase_uid=self.firebase_uid,
        )

    def _set_logger_to_verbose(self):
        # Use this for debugging
        self.config["logging.logging_level"] = "INFO"

    @track
    def send_dataset(
        self,
        dataset_id: str,
        receiver_project: str,
        receiver_api_key: str,
    ):
        """
        Send an individual a dataset. For this, you must know their API key.


        Parameters
        -----------

        dataset_id: str
            The name of the dataset
        receiver_project: str
            The project name that will receive the dataset
        receiver_api_key: str
            The project API key that will receive the dataset


        Example
        --------

        .. code-block::

            client = Client()
            client.send_dataset(
                dataset_id="research",
                receiver_project="...",
                receiver_api_key="..."
            )

        """
        return self.admin.send_dataset(
            dataset_id=dataset_id,
            receiver_project=receiver_project,
            receiver_api_key=receiver_api_key,
        )

    @track
    def clone_dataset(
        self,
        source_dataset_id: str,
        new_dataset_id: str = None,
        source_project: str = None,
        source_api_key: str = None,
        project: str = None,
        api_key: str = None,
    ):
        """
        Clone a dataset from another user's projects into your project.

        Parameters
        ----------
        dataset_id:
            The dataset to copy
        source_dataset_id:
            The original dataset
        source_project:
            The original project to copy from
        source_api_key:
            The original API key of the project
        project:
            The original project
        api_key:
            The original API key

        Example
        -----------

        .. code-block::

            client = Client()
            client.send_dataset(
                dataset_id="research",
                source_project="...",
                source_api_key="..."
            )

        """
        if source_api_key is None:
            source_api_key = self.api_key
            source_project = self.project

        if new_dataset_id is None:
            new_dataset_id = source_dataset_id

        return self.admin.copy_foreign_dataset(
            dataset_id=new_dataset_id,
            source_dataset_id=source_dataset_id,
            source_project=source_project,
            source_api_key=source_api_key,
            project=project,
            api_key=api_key,
        )

    @property
    def references(self):
        from relevanceai.__init__ import __version__

        REFERENCE_URL = f"https://relevanceai.readthedocs.io/en/{__version__}/"
        MESSAGE = f"You can find your references here {REFERENCE_URL}."
        print(MESSAGE)

    docs = references

    def search_app(self, dataset_id: str = None):
        if dataset_id is not None:
            self.print_search_dashboard_url(dataset_id)
        elif hasattr(self, "_dataset_id"):
            self.print_search_dashboard_url(self._dataset_id)
        elif hasattr(self, "dataset_id"):
            self.print_search_dashboard_url(self.dataset_id)
        else:
            print("You can build your search app at https://cloud.relevance.ai")
