import pytest

from relevanceai import Client


@pytest.fixture(scope="session", autouse=True)
def test_sample_dataset(test_client: Client, simple_doc, test_dataset_id):
    """Sample dataset to insert and then delete"""
    simple_documents = simple_doc * 1000
    response = test_client._insert_documents(test_dataset_id, simple_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)
