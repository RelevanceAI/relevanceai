import pytest
from relevanceai.http_client import Client, ClusterOps

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="session")
def kmeans_clusterer(test_client: Client, vector_dataset_id: str):
    df = test_client.Dataset(vector_dataset_id)
    clusterer: ClusterOps = df.auto_cluster("kmeans-20", vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(df.dataset_id, VECTOR_FIELDS)
