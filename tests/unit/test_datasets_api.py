from relevanceai.http_client import Dataset, Client


def test_datasets_api(test_dataset_df: Dataset):
    """Testing the datasets API
    Simple smoke tests for now until we are happy with functionality :)
    """
    test_dataset_df.info()
    test_dataset_df.describe()
    test_dataset_df.head()
    assert True


def test_info(test_dataset_df):
    info = test_dataset_df.info()
    assert True


def test_shape(test_dataset_df):
    shape = test_dataset_df.shape
    assert True


def test_head(test_dataset_df):
    head = test_dataset_df.head()
    assert True


def test_describe(test_dataset_df):
    description = test_dataset_df.describe()
    assert True


def test_cluster(test_dataset_df):
    centroids = test_dataset_df.cluster(field="sample_1_vector_", overwrite=True)
    assert True


def test_groupby_agg(test_dataset_df):
    agg = test_dataset_df.agg({"sample_1_label": "avg"})
    groupby_agg = test_dataset_df.groupby(["sample_1_description"]).agg(
        {"sample_1_label": "avg"}
    )
    assert True


def test_centroids(test_clustered_dataset_df):
    """Test a few functions at once."""
    # Abbreviate it so we can focus on the logic
    df = test_clustered_dataset_df
    closest = df.centroids(["sample_1_vector_"], "kmeans_10").closest()
    furthest = df.centroids(["sample_1_vector_"], "kmeans_10").furthest()
    agg = df.centroids(["sample_1_vector_"], "kmeans_10").agg({"sample_2_label": "avg"})
    groupby_agg = (
        df.centroids(["sample_1_vector_"], "kmeans_10")
        .groupby(["sample_3_description"])
        .agg({"sample_2_label": "avg"})
    )
    assert True


def test_sample(test_dataset_df):
    sample_n = test_dataset_df.sample(n=10)
    assert len(sample_n) == 10


def test_cluster(test_client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    df.cluster("sample_1_vector_", n_clusters=10, overwrite=True)
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health


def test_custom_cluster(test_client, test_sample_vector_dataset):
    from sklearn.cluster import KMeans

    kmeans = KMeans(n_clusters=2, random_state=0)
    df = test_client.Dataset(test_sample_vector_dataset)
    df.cluster("sample_1_vector_", clusterer=kmeans, overwrite=True)
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.default" in db_health


def test_custom_cluster_with_alias(test_client: Client, test_sample_vector_dataset):
    """
    Custom cluster with alias
    """
    from sklearn.cluster import KMeans

    ALIAS = "random"
    kmeans = KMeans(n_clusters=2, random_state=0)
    df: Dataset = test_client.Dataset(test_sample_vector_dataset)
    df.cluster("sample_1_vector_", clusterer=kmeans, overwrite=True, alias=ALIAS)
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert "_cluster_" in db_health
    assert f"_cluster_.sample_1_vector_.{ALIAS}" in db_health
