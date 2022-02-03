"""
Module for testing the auto clustering API
"""
import pytest
from relevanceai.http_client import Dataset, Client, ClusterOps

VECTOR_FIELDS = ["sample_1_vector_"]


@pytest.fixture(scope="session")
def minibatch_clusterer(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    clusterer: ClusterOps = df.auto_cluster(
        "minibatchkmeans-20", vector_fields=VECTOR_FIELDS
    )
    yield clusterer
    clusterer.delete_centroids(df.dataset_id, VECTOR_FIELDS)


def test_batch_clusterer_closest(minibatch_clusterer: ClusterOps):
    assert len(minibatch_clusterer.list_closest_to_center()) > 0


def test_batch_clusterer_centroids(minibatch_clusterer: ClusterOps):
    assert len(minibatch_clusterer.centroids) > 0


@pytest.fixture(scope="session")
def kmeans_clusterer(test_client: Client, test_sample_vector_dataset: Dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    clusterer: ClusterOps = df.auto_cluster("kmeans-20", vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(df.dataset_id, VECTOR_FIELDS)


def test_kmeans_closest(kmeans_clusterer: ClusterOps):
    """Batch K Means ClusterOps object"""
    assert len(kmeans_clusterer.list_closest_to_center()) > 0


def test_kmeans_centroids(kmeans_clusterer: ClusterOps):
    """K Means ClusterOps object"""
    assert len(kmeans_clusterer.centroids) > 0
