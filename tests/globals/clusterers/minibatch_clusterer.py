import pytest

from relevanceai.interfaces import Dataset

from .utils import VECTOR_FIELDS

ORIGINAL_ALIAS = "minibatchkmeans-3"


@pytest.fixture(scope="function")
def minibatch_clusterer(test_df: Dataset):
    clusterer = test_df.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def minibatch_subclusterer(test_df: Dataset):
    # Running batch k means after clustering
    ALIAS = "minibatchkmeans-4"
    clusterer = test_df.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    clusterer = test_df.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_df, ALIAS
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def kmeans_subclusterer(test_df: Dataset):
    # Running K Means after clustering
    ALIAS = "kmeans-4"
    clusterer = test_df.auto_cluster(ORIGINAL_ALIAS, vector_fields=VECTOR_FIELDS)
    clusterer = test_df.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias=ORIGINAL_ALIAS,
    )
    yield test_df, ALIAS
    # clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)
