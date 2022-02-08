import pytest

from relevanceai import Client

from tests.globals.utils import CLUSTER_DATASET_ID


@pytest.fixture(scope="session")
def sample_obj_dataset(test_client: Client, test_dataclass_documents, test_dataset_id):
    """
    Use this dataset if you just want vector
    """
    response = test_client._insert_documents(test_dataset_id, test_dataclass_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_clustered_dataset(test_client: Client, sample_vector_documents):
    """
    Use this test dataset if you want a dataset with clusters already.
    """
    test_client._insert_documents(CLUSTER_DATASET_ID, sample_vector_documents)
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=CLUSTER_DATASET_ID,
        vector_fields=["sample_1_vector_"],
        k=10,
        alias="kmeans_10",
        overwrite=True,
    )
    yield CLUSTER_DATASET_ID
    test_client.datasets.delete(CLUSTER_DATASET_ID)
