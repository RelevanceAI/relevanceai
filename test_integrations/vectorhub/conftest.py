import os

import pytest

from relevanceai import Client
from relevanceai.datasets import mock_documents

REGION = os.getenv("TEST_REGION")


@pytest.fixture(scope="session")
def test_project():
    if REGION == "us-east-1":
        return os.getenv("TEST_US_PROJECT")
    return os.getenv("TEST_PROJECT")


@pytest.fixture(scope="session")
def test_api_key():
    if REGION == "us-east-1":
        return os.getenv("TEST_US_API_KEY")
    return os.getenv("TEST_API_KEY")


@pytest.fixture(scope="session")
def test_firebase_uid():
    return "relevanceai-sdk-test-user"


@pytest.fixture(scope="session")
def test_client(test_project, test_api_key, test_firebase_uid):
    if REGION is None:
        client = Client(
            project=test_project, api_key=test_api_key, firebase_uid=test_firebase_uid
        )
    else:
        client = Client(
            project=test_project,
            api_key=test_api_key,
            firebase_uid=test_firebase_uid,
            region=REGION,
        )
    return client


@pytest.fixture(scope="function")
def test_dataset(test_client):
    docs = mock_documents(10)
    ds = test_client.Dataset("sample")
    ds.insert_documents(docs)
    yield ds
    ds.delete()
