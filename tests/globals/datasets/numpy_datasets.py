import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def numpy_dataset(
    test_client: Client, numpy_documents: List[Dict], test_dataset_id: str
):
    response = test_client._insert_documents(test_dataset_id, numpy_documents)

    yield response, len(numpy_documents)

    test_client.datasets.delete(test_dataset_id)
