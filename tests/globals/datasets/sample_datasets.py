import pytest

from typing import Dict, List

from relevanceai import Client

from tests.globals.constants import generate_dataset_id


@pytest.fixture(scope="function")
def sample_dataset_id(test_client: Client, simple_documents: List[Dict]):
    test_dataset_id = generate_dataset_id()

    test_client._insert_documents(test_dataset_id, simple_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
