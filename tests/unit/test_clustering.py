import pytest
import random

@pytest.fixture(scope="session")
def test_kmeans(test_client, test_sample_vector_dataset):
    test_client.kmeans_cluster(
        dataset_id= test_sample_vector_dataset,
        vector_fields= ["sample_1_vector_"]
    )
    db_health = test_client.datasets.monitor.health(test_sample_vector_dataset)
    assert '_cluster_' in db_health
    