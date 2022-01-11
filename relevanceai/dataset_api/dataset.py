"""
Pandas like dataset API
"""
import pandas as pd

from typing import List, Union

from relevanceai.api.client import BatchAPIClient
from relevanceai.vector_tools.cluster import Cluster


class Series(BatchAPIClient):
    """ """

    def __init__(self, project, api_key) -> None:
        """"""
        super().__init__(project, api_key)
        self.project = project
        self.api_key = api_key

    def __call__(self, dataset_id, field):
        """ """
        self.dataset_id = dataset_id
        self.field = field
        return self

    def vectorize(self, model):
        """ """
        if hasattr(model, "encode_documents"):

            def encode_documents(documents):
                return model.encode_documents(self.field, documents)

        else:

            def encode_documents(documents):
                return model([self.field], documents)

        self.pull_update_push(self.dataset_id, encode_documents)


class Dataset(Cluster):
    """ """

    def __init__(self, project, api_key) -> None:
        """ """
        super().__init__(project, api_key)
        self.project = project
        self.api_key = api_key

    def __call__(
        self,
        dataset_id: str,
        image_fields: List = [],
        text_fields: List = [],
        audio_fields: List = [],
        output_format: str = "pandas",
    ):
        """ """
        self.dataset_id = dataset_id
        self.image_fields = image_fields
        self.text_fields = text_fields
        self.audio_fields = audio_fields
        self.output_format = output_format

        return self

    @property
    def shape(self):
        """ """
        return self.get_number_of_documents(dataset_id=self.dataset_id)

    def __getitem__(self, field):
        """ """
        series = Series(self.project, self.api_key)
        series(dataset_id=self.dataset_id, field=field)
        return series

    def info(self) -> dict:
        """ """
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
            The first `n` rows of the caller object.
        """
        head_documents = self.get_documents(
            dataset_id=self.dataset_id,
            number_of_documents=n,
        )
        if raw_json:
            return head_documents
        else:
            return pd.DataFrame(head_documents).head(n=n)

    def describe(self) -> dict:
        """ """
        return self.datasets.facets(self.dataset_id)

    def vectorize(self, field, model):
        """ """
        series = Series(self.project, self.api_key)
        series(self.dataset_id, field).vectorize(model)

    def cluster(self, field, n_clusters):
        """ """
        centroids = self.kmeans_cluster(
            dataset_id = self.dataset_id, 
            vector_fields=[field],
            k = n_clusters,
            alias = f'kmeans_{n_clusters}'
        )
        return centroids
