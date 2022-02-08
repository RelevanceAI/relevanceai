import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def large_dataset(
    test_client: Client, sample_documents: List[Dict], test_dataset_id: str
):
    test_client._insert_documents(test_dataset_id, sample_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
