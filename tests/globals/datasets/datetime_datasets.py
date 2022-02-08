import pytest

from typing import Dict, List

from relevanceai import Client


@pytest.fixture(scope="session")
def datetime_dataset(
    test_client: Client, datetime_documents: List[Dict], test_dataset_id: str
):
    response = test_client._insert_documents(test_dataset_id, datetime_documents)

    yield response, len(datetime_documents)

    test_client.datasets.delete(test_dataset_id)
