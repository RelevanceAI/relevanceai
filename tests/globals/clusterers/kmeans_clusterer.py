import pytest

from relevanceai.dataset import Dataset

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="function")
def kmeans_clusterer(test_dataset: Dataset):
    clusterer = test_dataset.auto_cluster("kmeans-2", vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)
