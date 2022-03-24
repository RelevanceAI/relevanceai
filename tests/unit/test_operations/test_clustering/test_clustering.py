import pytest
import time
from typing import Dict, List

from sklearn.cluster import MiniBatchKMeans

from relevanceai.client import Client
from relevanceai.dataset import Dataset

from relevanceai.operations.cluster import ClusterOps
from relevanceai.operations.cluster.base import ClusterBase
from relevanceai.operations.cluster.groupby import ClusterGroupby

from tests.globals.constants import generate_random_string

CLUSTER_ALIAS = "kmeans_10"
VECTOR_FIELDS = ["sample_1_vector_"]


def test_kmeans(test_client: Client, clustered_dataset_id: List[Dict]):
    db_health = test_client.datasets.monitor.health(clustered_dataset_id)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans-10" in db_health


@pytest.fixture
def closest_to_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    time.sleep(2)
    results = test_client.datasets.cluster.centroids.list_closest_to_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans-10",
    )
    return results


@pytest.fixture
def furthest_from_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    time.sleep(2)
    results = test_client.datasets.cluster.centroids.list_furthest_from_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans-10",
    )
    return results


def test_furthest_different_from_closest(closest_to_centers, furthest_from_centers):
    """Ensure that the bug where they are closest and furthest are no longer there"""
    assert closest_to_centers != furthest_from_centers


@pytest.fixture
def test_clusterer(test_client: Client, clustered_dataset_id: Dataset):
    df: Dataset = test_client.Dataset(clustered_dataset_id)

    model = get_model()

    clusterer: ClusterOps = df.cluster(
        model=model, vector_fields=VECTOR_FIELDS, alias=CLUSTER_ALIAS
    )
    time.sleep(2)
    return clusterer


def get_model():
    # get a kmeans model
    from relevanceai.operations.cluster.models.kmeans import KMeansModel

    return KMeansModel(verbose=False)


def test_cluster(test_dataset: Dataset):

    vector_field = "sample_1_vector_"
    alias = "test_alias"

    model = get_model()

    test_dataset.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema


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


def test_agg_std(test_clusterer: ClusterOps):
    agg = test_clusterer.agg({"sample_2_value": "avg"})
    cluster_groupby: ClusterGroupby = test_clusterer.groupby(["sample_3_description"])
    groupby_agg = cluster_groupby.agg({"sample_2_value": "std_deviation"})
    assert isinstance(groupby_agg, dict)
    assert len(groupby_agg) > 0


def test_fit_predict(test_client: Client, vector_dataset_id: str):
    import random

    class CustomClusterModel(ClusterBase):
        def fit_predict(self, X):
            cluster_labels = [random.randint(0, 100) for _ in range(len(X))]
            return cluster_labels

    model = CustomClusterModel()

    df = test_client.Dataset(vector_dataset_id)
    clusterer = test_client.ClusterOps(
        alias="random_clustering",
        model=model,
    )
    clusterer.fit_predict(df, vector_fields=["sample_1_vector_"])
    assert "_cluster_.sample_1_vector_.random_clustering" in df.schema


def test_cluster(test_dataset: Dataset):
    from sklearn.cluster import KMeans

    vector_field = "sample_1_vector_"
    alias = generate_random_string().lower()

    # check they're not in first
    assert f"_cluster_.{vector_field}.{alias}" not in test_dataset.schema

    model = KMeans()
    clusterer = test_dataset.cluster(
        model=model, alias=alias, vector_fields=[vector_field], overwrite=True
    )
    assert f"_cluster_.{vector_field}.{alias}" in test_dataset.schema
    assert len(clusterer.list_closest_to_center()) > 0


def test_dbscan(test_client: Client, test_dataset: Dataset):
    from sklearn.cluster import DBSCAN

    ALIAS = "dbscan"

    model = DBSCAN()
    clusterer = test_client.ClusterOps(alias=ALIAS, model=model)
    clusterer.fit(test_dataset, ["sample_3_vector_"])
    assert any([x for x in test_dataset.schema if ALIAS in x])


@pytest.fixture(scope="function")
def test_batch_clusterer(test_client: Client, vector_dataset_id, test_dataset: Dataset):

    clusterer: ClusterOps = test_client.ClusterOps(
        alias=CLUSTER_ALIAS,
        model=MiniBatchKMeans(),
        dataset_id=vector_dataset_id,
        vector_fields=VECTOR_FIELDS,
    )

    clusterer.vector_fields = VECTOR_FIELDS
    closest = clusterer.list_closest_to_center(
        dataset=vector_dataset_id, vector_fields=VECTOR_FIELDS
    )
    assert len(closest["results"]) > 0

    clusterer.vector_fields = VECTOR_FIELDS
    furthest = clusterer.list_furthest_from_center(
        dataset=vector_dataset_id,
        vector_fields=VECTOR_FIELDS,
    )
    assert len(furthest["results"]) > 0

    df = test_client.Dataset(vector_dataset_id)
    clusterer.partial_fit_predict_update(
        dataset=df,
        vector_fields=VECTOR_FIELDS,
    )

    assert f"_cluster_.{VECTOR_FIELDS[0]}.{CLUSTER_ALIAS}" in df.schema
