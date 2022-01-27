import pytest

from sklearn.cluster import KMeans


def test_Dataset(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    assert True


def test_info(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    info = df.info()
    assert True


def test_shape(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    shape = df.shape
    assert True


def test_head(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    head = df.head()
    assert True


def test_describe(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    description = df.describe()
    assert True


def test_cluster(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    from relevanceai.clusterer.kmeans_clusterer import KMeansModel

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = KMeansModel()

    df.cluster(model=model, alias=alias, vector_fields=[vector_field], overwrite=True)
    assert f"_cluster_.{vector_field}.{alias}" in test_client.schema


def test_groupby_agg(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    agg = df.agg({"sample_1_label": "avg"})
    groupby_agg = df.groupby(["sample_1_description"]).agg({"sample_1_label": "avg"})
    assert True


def test_groupby_mean_method(test_client, test_dataset_df):
    manual_mean = test_dataset_df.groupby(["sample_1_label"]).agg(
        {"sample_1_value": "avg"}
    )

    assert manual_mean == test_dataset_df.groupby(["sample_1_label"]).mean(
        "sample_1_value"
    )


def test_centroids(test_client, test_clustered_dataset):
    df = test_client.Dataset(test_clustered_dataset)
    closest = df.centroids(["sample_1_vector_"], "kmeans_10").closest()
    furthest = df.centroids(["sample_1_vector_"], "kmeans_10").furthest()
    agg = df.centroids(["sample_1_vector_"], "kmeans_10").agg({"sample_2_label": "avg"})
    groupby_agg = (
        df.centroids(["sample_1_vector_"], "kmeans_10")
        .groupby(["sample_3_description"])
        .agg({"sample_2_label": "avg"})
    )
    assert True


def test_sample(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df.sample(n=10)
    assert len(sample_n) == 10


def test_series_sample(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df["sample_1_label"].sample(n=10)
    assert len(sample_n) == 10


def test_series_sample(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    sample_n = df[["sample_1_label", "sample_2_label"]].sample(n=10)
    assert len(sample_n) == 10
    assert len(sample_n[0].keys()) == 3


def test_value_counts(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    value_counts = df["sample_1_label"].value_counts()
    value_counts = df["sample_1_label"].value_counts(normalize=True)
    assert True


def test_filter(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    items = df.filter(items=["sample_1_label"])
    like = df.filter(like="sample_1_label")
    regex = df.filter(regex="s$")
    assert True
