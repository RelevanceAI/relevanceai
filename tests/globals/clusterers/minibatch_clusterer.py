import pytest

from relevanceai.dataset import Dataset

from .utils import VECTOR_FIELDS

ORIGINAL_ALIAS = "minibatchkmeans-3"


@pytest.fixture(scope="function")
def minibatch_clusterer(test_dataset: Dataset):
    clusterer = test_dataset.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def minibatch_subclusterer(test_dataset: Dataset):
    # Running batch k means after clustering
    ALIAS = "minibatchkmeans-4"
    clusterer = test_dataset.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    clusterer = test_dataset.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_df, ALIAS
    clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def kmeans_subclusterer(test_dataset: Dataset):
    # Running K Means after clustering
    ALIAS = "kmeans-4"
    clusterer = test_dataset.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    clusterer = test_dataset.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_df, ALIAS
    # clusterer.delete_centroids(test_dataset.dataset_id, VECTOR_FIELDS)
