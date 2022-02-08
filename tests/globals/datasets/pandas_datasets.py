import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def pandas_dataset(
    test_client: Client,
    pandas_documents: List[Dict],
    pandas_test_dataset_id: str,
):
    response = test_client._insert_documents(pandas_test_dataset_id, pandas_documents)

    yield response, len(pandas_documents)

    test_client.datasets.delete(pandas_test_dataset_id)
