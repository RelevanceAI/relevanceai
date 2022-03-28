# -*- coding: utf-8 -*-
"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

import time
from relevanceai import Client
from relevanceai.dataset import Dataset

from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.cluster.base import CentroidClusterBase


@pytest.mark.skip(reason="ClusterOps fit method missing")
def test_dataset_api_kmeans_integration(test_client: Client, test_dataset: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    clusterer = test_client.ClusterOps(model=KMeans(n_clusters=2), alias=alias)

    clusterer.fit(dataset=test_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema
