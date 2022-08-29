import pytest

from typing import Dict, List

from relevanceai import Client

from tests.globals.constants import generate_dataset_id


@pytest.fixture(scope="function")
def large_dataset_id(test_client: Client, sample_documents: List[Dict]):
    test_dataset_id = generate_dataset_id()

    test_dataset = test_client.Dataset(test_dataset_id)
    test_dataset.insert_documents(sample_documents)

    yield test_dataset_id

    test_client.datasets.delete(test_dataset_id)
