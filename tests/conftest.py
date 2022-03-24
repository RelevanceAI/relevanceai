"""All fixtures go here.
"""
import pytest
import pandas as pd
import tempfile

from relevanceai.client import Client
from relevanceai.dataset import Dataset

from tests.globals.constants import *
from tests.globals.document import *
from tests.globals.documents import *
from tests.globals.objects import *
from tests.globals.datasets import *
from tests.globals.clusterers import *
from tests.globals.constants import *

from tests.utils import *


@pytest.fixture(scope="session")
def test_token():
    return TEST_TOKEN


@pytest.fixture(scope="session")
def test_client(test_token):
    client = Client(token=test_token)
    client.config["mixpanel.is_tracking_enabled"] = False
    client.disable_analytics_tracking()
    yield client

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
