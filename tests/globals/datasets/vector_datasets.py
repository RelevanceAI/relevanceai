import time
import pytest

from typing import Dict, List

from relevanceai import Client

from tests.globals.constants import generate_dataset_id


@pytest.fixture(scope="function")
def vector_dataset_id(test_client: Client, vector_documents: List[Dict]):
    test_dataset_id = generate_dataset_id()
    test_dataset = test_client.Dataset(test_dataset_id)
    test_dataset.insert_documents(vector_documents)
    time.sleep(1)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)
