import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def vector_dataset(
    test_client: Client, vector_documents: List[Dict], test_dataset_id: str
):
    test_client._insert_documents(test_dataset_id, vector_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
