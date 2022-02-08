import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def large_sample_dataset(
    test_client: Client, test_sample_documents: List[Dict], test_dataset_id: str
):
    response = test_client._insert_documents(test_dataset_id, test_sample_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
