"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

from relevanceai import Client
from relevanceai.dataset_api import Dataset

from relevanceai.clusterer import Clusterer
from relevanceai.clusterer import CentroidClusterBase


def test_dataset_api_kmeans_integration(test_client: Client, test_dataset_df: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    class KMeansModel(CentroidClusterBase):
        def __init__(self, model):
            self.model: KMeans = model

        def fit_transform(self, vectors):
            return self.model.fit_predict(vectors)

        def get_centers(self):
            return self.model.cluster_centers_

    model = KMeansModel(model=KMeans())

    clusterer = test_client.Clusterer(model=model, alias=alias)

    clusterer.fit(dataset=test_dataset_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_dataset_df.schema
