import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def sample_dataset(test_client: Client, simple_documents, test_dataset_id):
    response = test_client._insert_documents(test_dataset_id, simple_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)
