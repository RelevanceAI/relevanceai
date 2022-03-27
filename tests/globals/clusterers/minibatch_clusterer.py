import pytest

from sklearn.cluster import KMeans, MiniBatchKMeans

from relevanceai.dataset import Dataset

from .utils import VECTOR_FIELDS

ORIGINAL_ALIAS = "minibatchkmeans-3"


@pytest.fixture(scope="function")
def minibatch_clusterer(test_dataset: Dataset):
    alias = "minibatchkmeans-3"
    clusterer = test_dataset.cluster(
        model=MiniBatchKMeans(n_clusters=3),
        alias=alias,
        vector_fields=VECTOR_FIELDS,
    )
    yield clusterer
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def minibatch_subclusterer(test_dataset: Dataset):
    alias = "minibatchkmeans-4"
    clusterer = test_dataset.cluster(
        model=MiniBatchKMeans(n_clusters=4),
        alias=alias,
        vector_fields=VECTOR_FIELDS,
    )
    clusterer = test_dataset.cluster(
        model=MiniBatchKMeans(n_clusters=4),
        alias=alias,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_dataset, alias
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def kmeans_subclusterer(test_dataset: Dataset):
    alias = "kmeans-4"
    clusterer = test_dataset.cluster(
        model=KMeans(n_clusters=4),
        alias=alias,
        vector_fields=VECTOR_FIELDS,
    )
    clusterer = test_dataset.cluster(
        model=KMeans(n_clusters=4),
        alias=alias,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_dataset, alias
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)
