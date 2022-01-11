"""
Pandas like dataset API
"""
import pandas as pd

from relevanceai.api.client import BatchAPIClient
from typing import List, Union


class Series:
    """
    A wrapper class for being able to vectorize documents over field
    """

    def __init__(self, client) -> None:
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

    def __init__(self, client) -> None:
        """
        Initialise the class
        """
        self.client = client

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
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
        self.output_format = output_format

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
        schema = self.client.datasets.schema(self.dataset_id)
        n_documents = self.client.get_number_of_documents(dataset_id=self.dataset_id)
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
        series = Series(self.client)
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
        health = self.client.datasets.monitor.health(self.dataset_id)
        schema = self.client.datasets.schema(self.dataset_id)
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

    def head(self, n: int = 5, raw_json: bool = False) -> Union[dict, pd.DataFrame]:
        """
        Return the first `n` rows.
        returns the first `n` rows of your dataset.
        It is useful for quickly testing if your object
        has the right type of data in it.

        Parameters
        ----------
        n : int, default 5
            Number of rows to select.

        Returns
        -------
        Pandas DataFrame or Dict, depending on args
            The first 'n' rows of the caller object.
        """
        head_documents = self.client.get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            return pd.DataFrame(head_documents).head(n=n)

    def describe(self) -> dict:
        """
        Descriptive statistics include those that summarize the central tendency
        dispersion and shape of a dataset's distribution, excluding NaN values.
        """
        return self.client.datasets.facets(self.dataset_id)

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
        series = Series(self.client)
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
        centroids = self.client.vector_tools.cluster.kmeans_cluster(
            dataset_id=self.dataset_id,
            vector_fields=[field],
            k=n_clusters,
            alias=f"kmeans_{n_clusters}",
            overwrite=overwrite,
        )
        return centroids


class Datasets(BatchAPIClient):
    """Dataset class for multiple datasets"""

    def __init__(self, client):
        self.client = client
