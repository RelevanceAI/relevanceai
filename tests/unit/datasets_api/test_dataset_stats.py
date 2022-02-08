"""
    Testing dataset
"""

from typing import Dict, List

from relevanceai.http_client import Dataset, Client


def test_cluster(test_client: Client, vector_dataset_id: str):
    df = test_client.Dataset(vector_dataset_id)
    from relevanceai.clusterer.kmeans_clusterer import KMeansModel

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = KMeansModel()

    df.cluster(model=model, alias=alias, vector_fields=[vector_field], overwrite=True)
    assert f"_cluster_.{vector_field}.{alias}" in df.schema


def test_centroids(test_client: Client, clustered_dataset: List[Dict]):
    df = test_client.Dataset(clustered_dataset)
    closest = df.centroids(["sample_1_vector_"], "kmeans_10").closest()
    furthest = df.centroids(["sample_1_vector_"], "kmeans_10").furthest()
    agg = df.centroids(["sample_1_vector_"], "kmeans_10").agg({"sample_2_label": "avg"})
    groupby_agg = (
        df.centroids(["sample_1_vector_"], "kmeans_10")
        .groupby(["sample_3_description"])
        .agg({"sample_2_label": "avg"})
    )
    assert True


def test_groupby_agg(test_dataset_df: Dataset):
    agg = test_dataset_df.agg({"sample_1_label": "avg"})
    groupby_agg = test_dataset_df.groupby(["sample_1_description"]).mean(
        "sample_1_label"
    )
    assert True


def test_groupby_agg_with_dict(test_sample_nested_assorted_dataset_df: Dataset):
    agg = test_sample_nested_assorted_dataset_df.agg({"sample_1": "avg"})
    groupby_agg = test_sample_nested_assorted_dataset_df.groupby(
        ["sample_1_description"]
    ).agg({"sample_1": "avg"})
    assert True


def test_groupby_mean_method(test_dataset_df: Dataset):
    manual_mean = test_dataset_df.groupby(["sample_1_label"]).agg(
        {"sample_1_value": "avg"}
    )

    assert manual_mean == test_dataset_df.groupby(["sample_1_label"]).mean(
        "sample_1_value"
    )


def test_smoke(test_dataset_df):
    test_dataset_df.health
    assert True
