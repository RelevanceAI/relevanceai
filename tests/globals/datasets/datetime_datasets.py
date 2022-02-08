import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def test_datetime_dataset(
    test_client: Client, sample_datetime_documents, test_dataset_id
):
    """Sample datetime dataset"""
    response = test_client._insert_documents(test_dataset_id, sample_datetime_documents)
    yield response, len(sample_datetime_documents)
    test_client.datasets.delete(test_dataset_id)
