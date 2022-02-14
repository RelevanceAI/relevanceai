"""
    Testing dataset
"""

from typing import Dict, List

from relevanceai.http_client import Dataset, Client


def test_cluster(test_df: Dataset):
    from relevanceai.clusterer.kmeans_clusterer import KMeansModel

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = KMeansModel()

    test_df.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in test_df.schema


def test_centroids(test_clustered_df: Dataset):
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans_10").closest()
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans_10").furthest()
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans_10").agg(
        {"sample_2_label": "avg"}
    )
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans_10").groupby(
        ["sample_3_description"]
    ).agg({"sample_2_label": "avg"})
    assert True


def test_groupby_agg(test_df: Dataset):
    test_df.agg({"sample_1_label": "avg"})
    test_df.groupby(["sample_1_description"]).mean("sample_1_label")
    assert True


def test_groupby_mean_method(test_df: Dataset):
    manual_mean = test_df.groupby(["sample_1_label"]).agg({"sample_1_value": "avg"})
    assert manual_mean == test_df.groupby(["sample_1_label"]).mean("sample_1_value")


def test_smoke(test_df: Dataset):
    test_df.health
    assert True
