import os
import pytest
from relevanceai import Client

@pytest.fixture
def test_project():
    # test projects
    return os.getenv("TEST_PROJECT")

@pytest.fixture
def test_api_key():
    return os.getenv("TEST_API_KEY")

@pytest.fixture
def simple_docs():
    return [{
        "_id": "id_1",
        "value": 10
    },
    {
        "_id": "id_2",
        "value": 20
    }]

@pytest.fixture
def test_client(test_project, test_api_key):
    return Client(test_project, test_api_key)

@pytest.fixture
def test_dataset_id():
    return "_sample_test_dataset"
