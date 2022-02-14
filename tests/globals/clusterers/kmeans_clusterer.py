import pytest

from relevanceai.http_client import Dataset

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="function")
def kmeans_clusterer(test_df: Dataset):
    clusterer = test_df.auto_cluster("kmeans-20", vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)
