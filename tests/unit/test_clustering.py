import pytest
import random


def test_kmeans(test_client, test_sample_vector_dataset):
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=["sample_1_vector_"],
        overwrite=True,
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    test_client.vector_tools.cluster.plot_clusters(
        test_sample_vector_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    metrics = test_client.vector_tools.cluster.cluster_metrics(
        test_sample_vector_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    distribution = test_client.vector_tools.cluster.cluster_distribution(
        test_sample_vector_dataset,
        "sample_1_vector_",
        "kmeans_10",
        ground_truth_field="sample_1_label",
    )
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.kmeans_10" in db_health
    assert True


def test_hdbscan_cluster(test_client, test_sample_vector_dataset):
    test_client.vector_tools.cluster.hdbscan_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=["sample_1_vector_"],
        overwrite=True,
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert "_cluster_" in db_health
    assert "_cluster_.sample_1_vector_.hdbscan" in db_health
