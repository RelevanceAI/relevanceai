import pytest
import random

def test_kmeans(test_client, test_sample_vector_dataset):
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id= test_sample_vector_dataset,
        vector_fields= ["sample_1_vector_"],
        overwrite = True
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert '_cluster_' in db_health
    assert '_cluster_.sample_1_vector_.kmeans' in db_health

def test_hdbscan_cluster(test_client, test_sample_vector_dataset):
    test_client.vector_tools.cluster.hdbscan_cluster(
        dataset_id= test_sample_vector_dataset,
        vector_fields= ["sample_1_vector_"],
        overwrite = True
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert '_cluster_' in db_health
    assert '_cluster_.sample_1_vector_.hdbscan' in db_health
