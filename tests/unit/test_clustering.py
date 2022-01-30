import pytest
from ..utils import generate_random_string


def test_kmeans(test_client, test_clustered_dataset):
    db_health = test_client.datasets.monitor.health(test_clustered_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health


def test_kmeans_dashboard(test_client, test_sample_vector_dataset):
    centroids = test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=["sample_1_vector_"],
        alias="kmeans_10",
        overwrite=True,
    )
    assert True


def test_cluster_plot(test_client, test_clustered_dataset):
    test_client.vector_tools.cluster.plot(
        test_clustered_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


def test_cluster_metrics(test_client, test_clustered_dataset):
    metrics = test_client.vector_tools.cluster.metrics(
        test_clustered_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


def test_cluster_distribution(test_client, test_clustered_dataset):
    distribution = test_client.vector_tools.cluster.distribution(
        test_clustered_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert True


def test_centroid_distances(test_client, test_clustered_dataset):
    centroid_distances = test_client.vector_tools.cluster.centroid_distances(
        test_clustered_dataset, "sample_1_vector_", "kmeans_10"
    )
    assert True


@pytest.fixture
def closest_to_centers(test_client, test_clustered_dataset):
    results = test_client.datasets.cluster.centroids.list_closest_to_center(
        test_clustered_dataset,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


@pytest.fixture
def furthest_from_centers(test_client, test_clustered_dataset):
    results = test_client.datasets.cluster.centroids.list_furthest_from_center(
        test_clustered_dataset,
        ["sample_1_vector_"],
        "kmeans_10",
    )
    return results


def test_furthest_different_from_closest(closest_to_centers, furthest_from_centers):
    """Ensure that the bug where they are closest and furthest are no longer there"""
    assert closest_to_centers != furthest_from_centers


@pytest.mark.skip(reason="Not fully implemented")
def test_hdbscan_cluster(test_client, test_sample_vector_dataset):
    test_client.vector_tools.cluster.hdbscan_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=["sample_1_vector_"],
        overwrite=True,
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.hdbscan" in db_health
