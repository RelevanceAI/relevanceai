import pytest

from sklearn.cluster import KMeans

from relevanceai import ClusterOps

from relevanceai.dataset import Dataset

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="function")
def kmeans_clusterer(test_dataset: Dataset):
    clusterer: ClusterOps = test_dataset.cluster(
        model=KMeans(n_clusters=2),
        alias="kmeans-2",
        vector_fields=VECTOR_FIELDS,
    )
    yield clusterer
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)
