from relevanceai.dataset import Dataset


def test_cluster(test_dataset: Dataset):
    from relevanceai.operations.cluster.models.kmeans import KMeansModel

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = KMeansModel()

    test_dataset.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema


def test_centroids(test_clustered_df: Dataset):
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").closest()
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").furthest()
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").agg(
        {"sample_2_label": "avg"}
    )
    test_clustered_df.centroids(["sample_1_vector_"], "kmeans-10").groupby(
        ["sample_3_description"]
    ).agg({"sample_2_label": "avg"})
    assert True


def test_groupby_agg(test_dataset: Dataset):
    test_dataset.agg({"sample_1_label": "avg"})
    test_dataset.groupby(["sample_1_description"]).mean("sample_1_label")
    assert True


def test_groupby_mean_method(test_dataset: Dataset):
    manual_mean = test_dataset.groupby(["sample_1_label"]).agg(
        {"sample_1_value": "avg"}
    )
    assert manual_mean == test_dataset.groupby(["sample_1_label"]).mean(
        "sample_1_value"
    )


def test_health(test_dataset: Dataset):
    import pandas as pd

    dataframe_output = test_dataset.health(output_format="dataframe")
    assert type(dataframe_output) == pd.DataFrame
    json_output = test_dataset.health(output_format="json")
    assert type(json_output) == dict
