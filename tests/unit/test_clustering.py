import pytest
import random

def test_kmeans(test_client, test_sample_vector_dataset):
    test_client.cluster.kmeans_cluster(
        dataset_id= test_sample_vector_dataset,
        vector_fields= ["sample_1_vector_"]
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert '_cluster_' in db_health
    assert '_cluster_.sample_1_vector_.default' in db_health
