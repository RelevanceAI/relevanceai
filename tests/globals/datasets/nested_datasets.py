import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def sample_nested_assorted_dataset(
    test_client: Client, sample_nested_assorted_documents, test_dataset_id
):
    """
    Use this dataset if you just want vector
    """
    response = test_client._insert_documents(
        test_dataset_id, sample_nested_assorted_documents
    )
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_nested_assorted_dataset(
    test_client: Client, sample_nested_assorted_documents, test_dataset_id
):
    """Sample nested assorted dataset"""
    response = test_client._insert_documents(
        test_dataset_id, sample_nested_assorted_documents
    )
    yield response, len(sample_nested_assorted_documents)
    test_client.datasets.delete(test_dataset_id)
