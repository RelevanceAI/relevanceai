"""
Pandas like dataset API
"""
import warnings
import pandas as pd
from relevanceai.dataset_api.groupby import Groupby, Agg
from relevanceai.dataset_api.centroids import Centroids
from typing import List, Union
import math

from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient


class Series:
    """
    A wrapper class for being able to vectorize documents over field
    """

    def __init__(self, client: BatchAPIClient) -> None:
        """
        Initialise the class
        """
        self.client = client

    def __call__(self, dataset_id: str, field: str):
        """
        Instaniates a Series

        Parameters
        ----------
        dataset_id : str
            The dataset_id of concern
        field : str
            The field within the dataset that you would like to select

        Returns
        -------
        Self
        """
        self.dataset_id = dataset_id
        self.field = field
        return self

    def vectorize(self, model) -> None:
        """
        Vectorises over a field give a model architecture

        Parameters
        ----------
        model : Machine learning model for vectorizing text`
            The dataset_id of concern
        """
        if hasattr(model, "encode_documents"):

            def encode_documents(documents):
                return model.encode_documents(self.field, documents)

        else:

            def encode_documents(documents):
                return model([self.field], documents)

        self.client.pull_update_push(self.dataset_id, encode_documents)


class Dataset(BatchAPIClient):
    """
    A Pandas Like datatset API for interacting with the RelevanceAI python package
    """

    def __init__(self, project: str, api_key: str) -> None:
        """
        Initialise the class
        """
        self.project = project
        self.api_key = api_key
        self.vector_tools = VectorTools(project=project, api_key=api_key)
        super().__init__(project=project, api_key=api_key)

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
        highlight_fields: dict = {},
        output_format: str = "pandas",
    ):
        """
        Instaniates a Dataset

        Parameters
        ----------
        dataset_id : str
            The dataset_id of concern
        image_fields : str
            The image_fields within the dataset that you would like to select
        text_fields : str
            The text_fields within the dataset that you would like to select
        audio_fields : str
            The audio_fields within the dataset that you would like to select
        output_format : str
            The output format of the dataset

        Returns
        -------
        Self
        """
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.highlight_fields = highlight_fields
        self.output_format = output_format
        self.groupby = Groupby(self.project, self.api_key, self.dataset_id)
        self.agg = Agg(self.project, self.api_key, self.dataset_id)
        self.centroids = Centroids(self.project, self.api_key, self.dataset_id)

        return self

    @property
    def shape(self):
        """
        Returns the shape (N x C) of a dataset
        N = number of samples in the Dataset
        C = number of columns in the Dataset

        Returns
        -------
        Tuple
            (N, C)
        """
        schema = self.datasets.schema(self.dataset_id)
        n_documents = self.get_number_of_documents(dataset_id=self.dataset_id)
        return (n_documents, len(schema))

    def __getitem__(self, field: str):
        """
        Returns a Series Object that selects a particular field within a dataset

        Parameters
        ----------
        field : str
            the particular field within the dataset

        Returns
        -------
        Tuple
            (N, C)
        """
        series = Series(self)
        series(dataset_id=self.dataset_id, field=field)
        return series

    def info(self) -> dict:
        """
        Return a dictionary that contains information about the Dataset
        including the index dtype and columns and non-null values.

        Returns
        -------
        Dict
            Dictionary of information
        """
        health = self.datasets.monitor.health(self.dataset_id)
        schema = self.datasets.schema(self.dataset_id)
        schema = {key: str(value) for key, value in schema.items()}
        info = {
            key: {
                "Non-Null Count": health[key]["missing"],
                "Dtype": schema[key],
            }
            for key in schema.keys()
        }
        dtypes = {
            dtype: list(schema.values()).count(dtype)
            for dtype in set(list(schema.values()))
        }
        info = {"info": info, "dtypes": dtypes}
        return info

    def head(
        self, n: int = 5, raw_json: bool = False, **kw
    ) -> Union[dict, pd.DataFrame]:
        """
        Return the first `n` rows.
        returns the first `n` rows of your dataset.
        It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.
        raw_json: bool
            If True, returns raw JSON and not Pandas Dataframe
        kw:
            Additional arguments to feed into show_json

        Returns
        -------
        Pandas DataFrame or Dict, depending on args
            The first 'n' rows of the caller object.
        """
        head_documents = self.get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            try:
                return self._show_json(head_documents, **kw)
            except Exception as e:
                warnings.warn("Displaying using Pandas." + str(e))
                return pd.json_normalize(head_documents).head(n=n)

    def _show_json(self, docs, **kw):
        from jsonshower import show_json

        if not self.text_fields:
            text_fields = pd.json_normalize(docs).columns.tolist()
        else:
            text_fields = self.text_fields
        return show_json(
            docs,
            image_fields=self.image_fields,
            audio_fields=self.audio_fields,
            highlight_fields=self.highlight_fields,
            text_fields=text_fields,
        )

    def describe(self) -> dict:
        """
        Descriptive statistics include those that summarize the central tendency
        dispersion and shape of a dataset's distribution, excluding NaN values.
        """
        return self.datasets.facets(self.dataset_id)

    def vectorize(self, field, model):
        """
        Vectorizes a Particular field (text) of the dataset

        Parameters
        ----------
        field : str
            The text field to select
        model
            a Type deep learning model that vectorizes text
        """
        series = Series(self)
        series(self.dataset_id, field).vectorize(model)

    def cluster(self, field, n_clusters=10, overwrite=False):
        """
        Performs KMeans Clustering on over a vector field within the dataset.

        Parameters
        ----------
        field : str
            The text field to select
        n_cluster: int default = 10
            the number of cluster to find wihtin the vector field
        """
        centroids = self.vector_tools.cluster.kmeans_cluster(
            dataset_id=self.dataset_id,
            vector_fields=[field],
            k=n_clusters,
            alias=f"kmeans_{n_clusters}",
            overwrite=overwrite,
        )
        return centroids

    def sample(
        self,
        n: int = 0,
        frac: float = None,
        filters: list = [],
        random_state: int = 0,
    ):

        """
        Return a random sample of items from a dataset.

        Parameters
        ----------
        n : int
            Number of items to return. Cannot be used with frac.
        frac: float
            Fraction of items to return. Cannot be used with n.
        filters: list
            Query for filtering the search results
        random_state: int
            Random Seed for retrieving random documents.

        """
        if n == 0 and frac is None:
            raise ValueError("Must provide one of n or frac")

        if frac and n:
            raise ValueError("Only one of n or frac can be provided")

        if frac:
            if frac > 1 or frac < 0:
                raise ValueError("Fraction must be between 0 and 1")
            n = math.ceil(
                self.get_number_of_documents(self.dataset_id, filters=filters) * frac
            )

        return self.datasets.documents.get_where(
            dataset_id=self.dataset_id,
            filters=filters,
            page_size=n,
            random_state=random_state,
            is_random=True,
        )["documents"]


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, project: str, api_key: str):
        self.project = project
        self.api_key = api_key
        super().__init__(project=project, api_key=api_key)
