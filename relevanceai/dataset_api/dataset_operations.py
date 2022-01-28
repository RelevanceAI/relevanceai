"""
Pandas like dataset API
"""
import re
import math
import warnings
import pandas as pd
import numpy as np

from doc_utils import DocUtils

from typing import Dict, List, Union, Callable, Optional

from relevanceai.dataset_api.groupby import Groupby, Agg
from relevanceai.dataset_api.centroids import Centroids
from relevanceai.dataset_api.helpers import _build_filters

from relevanceai.vector_tools.client import VectorTools
from relevanceai.api.client import BatchAPIClient
from relevanceai.dataset_api.dataset_write import Write
from relevanceai.dataset_api.dataset_series import Series


class Operations(Write):
    def vectorize(self, field, model):
        """
        Vectorizes a Particular field (text) of the dataset

        Parameters
        ----------
        field : str
            The text field to select
        model
            a Type deep learning model that vectorizes text

        Example
        -------
        .. code-block::

            from relevanceai import Client
            from vectorhub.encoders.text.sentence_transformers import SentenceTransformer2Vec

            model = SentenceTransformer2Vec("all-mpnet-base-v2 ")

            client = Client()

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            text_field = "text_field"
            df.vectorize(text_field, model)
        """
        series = Series(self)
        series(self.dataset_id, field).vectorize(model)

    def cluster(self, model, alias, vector_fields, **kwargs):
        """
        Performs KMeans Clustering on over a vector field within the dataset.

        Parameters
        ----------
        model : Class
            The clustering model to use
        vector_fields : str
            The vector fields over which to cluster

        Example
        -------
        .. code-block::

            from relevanceai import Client
            from relevanceai.clusterer import Clusterer
            from relevanceai.clusterer.kmeans_clusterer import KMeansModel

            client = Client()

            dataset_id = "sample_dataset"
            df = client.Dataset(dataset_id)

            vector_field = "vector_field_"
            n_clusters = 10

            model = KMeansModel(k=n_clusters)

            df.cluster(model=model, alias=f"kmeans-{n_clusters}", vector_fields=[vector_field])
        """

        from relevanceai.clusterer import Clusterer

        clusterer = Clusterer(
            model=model, alias=alias, api_key=self.api_key, project=self.project
        )
        clusterer.fit(dataset=self, vector_fields=vector_fields)
        return clusterer
