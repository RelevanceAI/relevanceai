"""
    Testing dataset
"""

import pandas as pd
from relevanceai.http_client import Dataset, Client, Clusterer


def test_cluster(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    from relevanceai.clusterer.kmeans_clusterer import KMeansModel

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = KMeansModel()

    df.cluster(model=model, alias=alias, vector_fields=[vector_field], overwrite=True)
    assert f"_cluster_.{vector_field}.{alias}" in df.schema


def test_centroids(test_client, test_clustered_dataset: Dataset):
    CLUSTER_ALIAS = "kmeans_10"
    VECTOR_FIELDS = ["sample_1_vector_"]

    df: Dataset = test_client.Dataset(test_clustered_dataset)
    clusterer: Clusterer = df.cluster(VECTOR_FIELDS, CLUSTER_ALIAS)
    closest = clusterer.list_closest_to_center()
    furthest = clusterer.list_furthest_from_center()
    agg = clusterer.aggregate(VECTOR_FIELDS, CLUSTER_ALIAS).agg(
        {"sample_2_label": "avg"}
    )
    groupby_agg = clusterer.groupby(["sample_3_description"]).agg(
        {"sample_2_label": "avg"}
    )
    assert True
