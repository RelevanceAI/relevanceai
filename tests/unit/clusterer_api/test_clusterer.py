"""
    Testing dataset
"""

import pandas as pd
import pytest
from relevanceai.clusterer import kmeans_clusterer
from relevanceai.http_client import Dataset, Client, ClusterOps
from relevanceai.dataset_api.cluster_groupby import ClusterGroupby

CLUSTER_ALIAS = "kmeans_10"
VECTOR_FIELDS = ["sample_1_vector_"]


@pytest.fixture
def test_clusterer(test_client: Client, test_clustered_dataset: Dataset):
    df: Dataset = test_client.Dataset(test_clustered_dataset)

    model = get_model()

    clusterer: ClusterOps = df.cluster(
        model=model, vector_fields=VECTOR_FIELDS, alias=CLUSTER_ALIAS
    )
    return clusterer


def get_model():
    # get a kmeans model
    from relevanceai.clusterer.kmeans_clusterer import KMeansModel

    return KMeansModel(verbose=False)


def test_cluster(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = get_model()

    df.cluster(model=model, alias=alias, vector_fields=[vector_field], overwrite=True)
    assert f"_cluster_.{vector_field}.{alias}" in df.schema


def test_closest(test_clusterer: ClusterOps):
    closest = test_clusterer.list_closest_to_center()
    assert len(closest["results"]) > 0


def test_furthest(test_clusterer: ClusterOps):
    furthest = test_clusterer.list_furthest_from_center()
    assert len(furthest["results"]) > 0


def test_agg(test_clusterer: ClusterOps):
    agg = test_clusterer.agg({"sample_2_value": "avg"})
    cluster_groupby: ClusterGroupby = test_clusterer.groupby(["sample_3_description"])
    groupby_agg = cluster_groupby.agg({"sample_2_value": "avg"})
    assert isinstance(groupby_agg, dict)
    assert len(groupby_agg) > 0
