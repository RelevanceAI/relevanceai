"""All fixtures go here.
"""
import os
import pytest

import pandas as pd

from relevanceai import Client
from relevanceai.dataset_api import Dataset

import tempfile

from tests.globals.utils import *
from tests.globals.document import *
from tests.globals.documents import *
from tests.globals.objects import *
from tests.globals.datasets import *


@pytest.fixture(scope="session")
def test_project():
    # test projects
    return os.getenv("TEST_PROJECT")


@pytest.fixture(scope="session")
def test_api_key():
    return os.getenv("TEST_API_KEY")


@pytest.fixture(scope="session")
def test_client(test_project, test_api_key):
    client = Client(test_project, test_api_key)
    return client


@pytest.fixture(scope="session")
def test_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + RANDOM_DATASET_SUFFIX


@pytest.fixture(scope="session")
def pandas_test_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + RANDOM_PANDAS_DATASET_SUFFIX


@pytest.fixture(scope="session")
def test_csv_dataset(test_client: Client, sample_vector_documents, test_dataset_id):
    """Sample csv dataset"""

    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(sample_vector_documents)
        df.to_csv(csvfile)

        response = test_client._insert_csv(test_dataset_id, csvfile.name)
        yield response, len(sample_vector_documents)
        test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_read_df(test_client: Client, sample_vector_documents):
    DATASET_ID = "_sample_df_"
    df = test_client.Dataset(DATASET_ID)
    results = df.upsert_documents(sample_vector_documents)
    yield results
    df.delete()


@pytest.fixture(scope="session")
def test_csv_df(test_dataset_df: Dataset, sample_vector_documents, test_dataset_id):
    """Sample csv dataset"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(sample_vector_documents)
        df.to_csv(csvfile)

        response = test_dataset_df.insert_csv(csvfile.name)
        yield response, len(sample_vector_documents)
        test_dataset_df.delete()
