"""All fixtures go here.
"""
import os
import pytest
import pandas as pd
import tempfile
from relevanceai.interfaces import Dataset, Client
from tests.globals.constants import *
from tests.globals.document import *
from tests.globals.documents import *
from tests.globals.objects import *
from tests.globals.datasets import *
from tests.globals.clusterers import *
from tests.globals.constants import SAMPLE_DATASET_DATASET_PREFIX

REGION = os.getenv("TEST_REGION")


# def pytest_sessionstart(session):
#     """
#     Pytest's configuration
#     """
#     # Deleting all mixpanel analytics from tests
#     CONFIG_FN = os.path.join("relevanceai", "config.ini")
#     with open(CONFIG_FN, "r") as f:
#         lines = f.readlines()

#     os.remove(CONFIG_FN)

#     with open(CONFIG_FN, "w") as f:
#         for i, line in enumerate(lines):
#             if i < 27:
#                 f.write(line)


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


def correct_client_config(client):
    client.config.reset()
    if client.region != "us-east-1":
        raise ValueError("default value aint RIGHT")


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
    # For some reason not resetting to default
    # correct_client_config(client)
    client.config["mixpanel.is_tracking_enabled"] = False
    client.disable_analytics_tracking()
    yield client

    # To avoid flooding backend
    for d in client.list_datasets()["datasets"]:
        if SAMPLE_DATASET_DATASET_PREFIX in d:
            client.delete_dataset(d)


@pytest.fixture(scope="module")
def test_csv_dataset(test_client: Client, vector_documents: List[Dict]):
    correct_client_config(test_client)
    test_dataset_id = generate_dataset_id()

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(vector_documents)
        df.to_csv(csvfile)

        response = test_client._insert_csv(test_dataset_id, csvfile.name)
        yield response, len(vector_documents)
        test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="module")
def test_read_df(test_client: Client, vector_documents: List[Dict]):
    correct_client_config(test_client)
    DATASET_ID = generate_dataset_id()
    df = test_client.Dataset(DATASET_ID)
    test_client.disable_analytics_tracking()
    results = df.upsert_documents(vector_documents)
    yield results
    df.delete()


@pytest.fixture(scope="module")
def test_csv_df(test_df: Dataset, vector_documents: List[Dict]):
    """Sample csv dataset"""
    test_df.config.reset()
    test_client.disable_analytics_tracking()
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(vector_documents)
        df.to_csv(csvfile)

        response = test_df.insert_csv(csvfile.name)
        yield response, len(vector_documents)
