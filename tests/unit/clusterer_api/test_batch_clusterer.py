# -*- coding: utf-8 -*-
"""
    Testing Batch CLustering
"""

import pandas as pd
import pytest
from relevanceai.clusterer import kmeans_clusterer
from relevanceai.http_client import Dataset, Client, ClusterOps
from relevanceai.dataset_api.cluster_groupby import ClusterGroupby

CLUSTER_ALIAS = "minibatch"
VECTOR_FIELDS = ["sample_1_vector_"]


@pytest.fixture(scope="session")
def test_batch_clusterer(test_client: Client, test_sample_vector_dataset: Dataset):
    from sklearn.cluster import MiniBatchKMeans

    clusterer = test_client.ClusterOps(
        alias=CLUSTER_ALIAS,
        model=MiniBatchKMeans(),
        dataset_id=test_sample_vector_dataset,
        vector_fields=VECTOR_FIELDS,
    )
    yield clusterer
    clusterer.delete_centroids(test_sample_vector_dataset, VECTOR_FIELDS)


def test_cluster(
    test_batch_clusterer: ClusterOps,
    test_sample_vector_dataset: Dataset,
    test_client: Client,
):
    df = test_client.Dataset(test_sample_vector_dataset)
    vector_field = "sample_1_vector_"
    alias = CLUSTER_ALIAS

    test_batch_clusterer.partial_fit_predict_update(
        dataset=df,
        vector_fields=[vector_field],
    )
    assert f"_cluster_.{vector_field}.{alias}" in df.schema


def test_closest(test_batch_clusterer: ClusterOps, test_sample_vector_dataset: Dataset):
    test_batch_clusterer.vector_fields = VECTOR_FIELDS
    closest = test_batch_clusterer.list_closest_to_center(
        dataset=test_sample_vector_dataset, vector_fields=VECTOR_FIELDS
    )
    assert len(closest["results"]) > 0


def test_furthest(
    test_batch_clusterer: ClusterOps, test_sample_vector_dataset: Dataset
):
    test_batch_clusterer.vector_fields = VECTOR_FIELDS
    furthest = test_batch_clusterer.list_furthest_from_center(
        dataset=test_sample_vector_dataset,
        vector_fields=VECTOR_FIELDS,
    )
    assert len(furthest["results"]) > 0
