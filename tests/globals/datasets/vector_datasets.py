import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def test_sample_vector_dataset(
    test_client: Client, sample_vector_documents, test_dataset_id
):
    """
    Use this dataset if you just want vector
    """
    response = test_client._insert_documents(test_dataset_id, sample_vector_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)
