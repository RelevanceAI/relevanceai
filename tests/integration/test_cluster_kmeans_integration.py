# -*- coding: utf-8 -*-
"""
Test the clustering workflow from getting the documents, clustering and then inserting the relevant centroids
"""

import pytest

import time
from relevanceai import Client
from relevanceai.interfaces import Dataset

from relevanceai.workflows.cluster_ops.ops import ClusterOps
from relevanceai.workflows.cluster_ops.base import CentroidClusterBase


def test_dataset_api_kmeans_integration(test_client: Client, test_df: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    clusterer = test_client.ClusterOps(model=KMeans(n_clusters=2), alias=alias)

    clusterer.fit_predict_update(dataset=test_df, vector_fields=[vector_field])

    assert f"_cluster_.{vector_field}.{alias}" in test_df.schema
