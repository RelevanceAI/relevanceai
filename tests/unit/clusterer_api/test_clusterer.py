"""
    Testing dataset
"""

import pandas as pd
from relevanceai.clusterer import kmeans_clusterer
from relevanceai.http_client import Dataset, Client, Clusterer


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


def test_centroids(test_client, test_clustered_dataset: Dataset):
    CLUSTER_ALIAS = "kmeans_10"
    VECTOR_FIELDS = ["sample_1_vector_"]

    df: Dataset = test_client.Dataset(test_clustered_dataset)

    model = get_model()

    clusterer: Clusterer = df.cluster(
        model=model, vector_fields=VECTOR_FIELDS, alias=CLUSTER_ALIAS
    )
    closest = clusterer.list_closest_to_center()
    furthest = clusterer.list_furthest_from_center()
    agg = clusterer.agg({"sample_2_label": "avg"})
    groupby_agg = clusterer.groupby(["sample_3_description"]).agg(
        {"sample_2_label": "avg"}
    )
    assert True
