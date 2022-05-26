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
    return client


@pytest.fixture(scope="session", autouse=True)
def remove_test_datasets(test_client: Client):
    for dataset in test_client.list_datasets()["datasets"]:
        if SAMPLE_DATASET_DATASET_PREFIX in dataset:
            test_client.delete_dataset(dataset)


@pytest.fixture(scope="module")
def test_csv_dataset(test_client: Client, vector_documents: List[Dict]):
    test_dataset_id = generate_dataset_id()

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(vector_documents)
        df.to_csv(csvfile)

        response = test_client._insert_csv(test_dataset_id, csvfile.name)
        yield response, len(vector_documents)


@pytest.fixture(scope="module")
def test_read_df(test_client: Client, vector_documents: List[Dict]):
    DATASET_ID = generate_dataset_id()
    df = test_client.Dataset(DATASET_ID)
    results = df.upsert_documents(vector_documents)
    yield results


@pytest.fixture(scope="module")
def test_csv_df(test_dataset: Dataset, vector_documents: List[Dict]):
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(vector_documents)
        df.to_csv(csvfile)

        response = test_dataset.insert_csv(csvfile.name)
        yield response, len(vector_documents)


@pytest.fixture
def sample_document():
    return {"value": 100, "blast32": 21}


@pytest.fixture
def sample_nested_document():
    return {"value1": {"value2": {"value3": 100}}, "blast32": 21}


@pytest.fixture
def sample_2_document():
    return {"check_value": random.randint(0, 100)}


@pytest.fixture
def combined_sample_document():
    return {
        "value": random.randint(0, 99999),
        "check_value": random.randint(0, 9999999),
    }


@pytest.fixture
def sample_3_document():
    return {"check": random.randint(0, 100)}


@pytest.fixture
def sample_documents(sample_document, sample_nested_document):
    documents = [sample_document] * 50 + [sample_nested_document] * 50
    random.shuffle(documents)
    return documents
