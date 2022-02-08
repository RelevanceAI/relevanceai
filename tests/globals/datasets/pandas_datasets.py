import pytest

from relevanceai import Client


@pytest.fixture(scope="session")
def test_pandas_dataset(
    test_client: Client, sample_pandas_documents, pandas_test_dataset_id
):
    response = test_client._insert_documents(
        pandas_test_dataset_id, sample_pandas_documents
    )
    yield response, len(sample_pandas_documents)
    test_client.datasets.delete(pandas_test_dataset_id)
