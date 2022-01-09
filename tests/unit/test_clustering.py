import pytest
import random


def test_kmeans(test_client, test_clustered_dataset):
    db_health = test_client.datasets.monitor.health(test_clustered_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health


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
