import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def test_numpy_dataset(test_client: Client, sample_numpy_documents, test_dataset_id):
    response = test_client._insert_documents(test_dataset_id, sample_numpy_documents)
    yield response, len(sample_numpy_documents)
    test_client.datasets.delete(test_dataset_id)
