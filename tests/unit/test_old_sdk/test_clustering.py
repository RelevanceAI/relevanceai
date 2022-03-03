import pytest
import time
from typing import Dict, List

from relevanceai.http_client import Client


def test_kmeans(test_client: Client, clustered_dataset_id: List[Dict]):
    db_health = test_client.datasets.monitor.health(clustered_dataset_id)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health


def test_kmeans_dashboard(test_client: Client, vector_dataset_id: str):
    centroids = test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=vector_dataset_id,
        vector_fields=["sample_1_vector_"],
        alias="kmeans_10",
        overwrite=True,
    )
    assert True


def test_cluster_plot(test_client: Client, clustered_dataset_id: List[Dict]):
    test_client.vector_tools.cluster.plot(
        clustered_dataset_id,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


def test_cluster_metrics(test_client: Client, clustered_dataset_id: List[Dict]):
    metrics = test_client.vector_tools.cluster.metrics(
        clustered_dataset_id,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


def test_cluster_distribution(test_client: Client, clustered_dataset_id: List[Dict]):
    distribution = test_client.vector_tools.cluster.distribution(
        clustered_dataset_id,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


@pytest.mark.skip(reason="not rerouted lol")
def test_centroid_distances(test_client: Client, clustered_dataset_id: List[Dict]):
    centroid_distances = test_client.vector_tools.cluster.centroid_distances(
        clustered_dataset_id, "sample_1_vector_", "kmeans_10"
    )
    assert True


@pytest.fixture
def closest_to_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    results = test_client.datasets.cluster.centroids.list_closest_to_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


@pytest.fixture
def furthest_from_centers(test_client: Client, clustered_dataset_id: List[Dict]):
    results = test_client.datasets.cluster.centroids.list_furthest_from_center(
        clustered_dataset_id,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


def test_furthest_different_from_closest(closest_to_centers, furthest_from_centers):
    """Ensure that the bug where they are closest and furthest are no longer there"""
    time.sleep(2)
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
