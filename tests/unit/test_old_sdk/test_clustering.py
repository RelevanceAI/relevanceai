import pytest
import time
from typing import Dict, List

from relevanceai.interfaces import Client


def test_kmeans(test_client: Client, clustered_dataset_id: List[Dict]):
    db_health = test_client.datasets.monitor.health(clustered_dataset_id)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health


def test_kmeans_dashboard(test_client: Client, vector_dataset_id: str):
    ds = test_client.Dataset(vector_dataset_id)
    ds.auto_cluster("kmeans_10", ["sample_1_vector_"])
    assert True


@pytest.fixture
def closest_to_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    time.sleep(2)
    results = test_client.datasets.cluster.centroids.list_closest_to_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


@pytest.fixture
def furthest_from_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    time.sleep(2)
    results = test_client.datasets.cluster.centroids.list_furthest_from_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


def test_furthest_different_from_closest(closest_to_centers, furthest_from_centers):
    """Ensure that the bug where they are closest and furthest are no longer there"""
    assert closest_to_centers != furthest_from_centers


@pytest.mark.skip(reason="Not fully implemented")
def test_hdbscan_cluster(test_client: Client, vector_dataset_id: str):
    test_client.vector_tools.cluster.hdbscan_cluster(
        dataset_id=vector_dataset_id,
        vector_fields=["sample_1_vector_"],
        overwrite=True,
    )
    db_health = test_client.datasets.monitor.health(vector_dataset_id)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.hdbscan" in db_health
