import pytest

from relevanceai.http_client import Dataset

from .utils import VECTOR_FIELDS


@pytest.fixture(scope="session")
def kmeans_clusterer(test_df: Dataset):
    clusterer = test_df.auto_cluster("kmeans-20", vector_fields=VECTOR_FIELDS)
    return clusterer
