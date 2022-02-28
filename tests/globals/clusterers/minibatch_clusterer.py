import pytest

from relevanceai.http_client import Dataset

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="function")
def minibatch_clusterer(test_df: Dataset):
    clusterer = test_df.auto_cluster("minibatchkmeans-3", vector_fields=VECTOR_FIELDS)
    yield clusterer
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def minibatch_subclusterer(test_df: Dataset):
    # Running batch k means after clustering
    ALIAS = "minibatchkmeans-4"
    clusterer = test_df.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias="minibatchkmeans-3",
    )
    yield test_df, ALIAS
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)


@pytest.fixture(scope="function")
def kmeans_subclusterer(test_df: Dataset):
    # Running K Means after clustering
    ALIAS = "kmeans-4"
    clusterer = test_df.auto_cluster(
        ALIAS,
        vector_fields=VECTOR_FIELDS,
        parent_alias="minibatchkmeans-3",
    )
    yield test_df, ALIAS
    clusterer.delete_centroids(test_df.dataset_id, VECTOR_FIELDS)
