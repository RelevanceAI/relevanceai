"""All fixtures go here.
"""
import os
import pytest
import uuid
import random
import numpy as np
from relevanceai import Client
import datetime
import pandas as pd

from utils import generate_random_string, generate_random_vector, generate_random_label

RANDOM_STRING = str(random.randint(0, 999))


@pytest.fixture(scope="session")
def test_project():
    # test projects
    return os.getenv("TEST_PROJECT")


@pytest.fixture(scope="session")
def test_api_key():
    return os.getenv("TEST_API_KEY")


@pytest.fixture(scope="session")
def simple_doc():
    return [
        {
            "_id": uuid.uuid4().__str__(),
            "value": random.randint(0, 1000),
            "sample_1_vector_": generate_random_vector(N=100),
        }
    ]


@pytest.fixture(scope="session")
def test_client(test_project, test_api_key):
    return Client(test_project, test_api_key)


@pytest.fixture(scope="session")
def test_dataset_id():
    return "_sample_test_dataset" + RANDOM_STRING


@pytest.fixture(scope="session")
def test_sample_dataset(test_client, simple_doc, test_dataset_id):
    """Sample dataset to insert and then delete"""
    simple_docs = simple_doc * 1000
    response = test_client.insert_documents(test_dataset_id, simple_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_large_sample_dataset(test_client, simple_doc, test_dataset_id):
    """Sample dataset to insert and then delete"""
    simple_docs = simple_doc * 1000
    response = test_client.insert_documents(test_dataset_id, simple_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def error_doc():
    return [{"_id": 3, "value": np.nan}]


@pytest.fixture(scope="session")
def sample_vector_docs():
    def _sample_vector_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_label": generate_random_label(),
            "sample_2_label": generate_random_label(),
            "sample_3_label": generate_random_label(),
            "sample_1_description": generate_random_string(),
            "sample_2_description": generate_random_string(),
            "sample_3_description": generate_random_string(),
            "sample_1_vector_": generate_random_vector(N=100),
            "sample_2_vector_": generate_random_vector(N=100),
            "sample_3_vector_": generate_random_vector(N=100),
        }

    N = 100
    return [_sample_vector_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session")
def sample_datetime_docs():
    def _sample_datetime_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_datetime": datetime.now(),
            "sample_2_datetime": datetime.now(),
        }

    N = 20
    return [_sample_datetime_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session")
def sample_numpy_docs():
    def _sample_numpy_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_numpy": np.random.randint(5, size=1)[0],
            "sample_2_numpy": np.random.rand(3, 2),
        }

    N = 20
    return [_sample_numpy_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session")
def sample_pandas_docs():
    def _sample_numpy_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_pandas": pd.DataFrame(
                np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
            ),
            "sample_2_pandas": pd.DataFrame(
                np.random.randint(0, 10, size=(10, 4)), columns=list("ABCD")
            ),
        }

    N = 20
    return [_sample_numpy_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session")
def test_sample_vector_dataset(test_client, sample_vector_docs, test_dataset_id):
    """Sample vector dataset"""
    response = test_client.insert_documents(test_dataset_id, sample_vector_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_clustered_dataset(test_client, test_sample_vector_dataset):
    """Sample vector dataset"""
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=test_sample_vector_dataset,
        vector_fields=["sample_1_vector_"],
        overwrite=True,
    )
    yield test_sample_vector_dataset


@pytest.fixture(scope="session")
def test_datetime_dataset(test_client, sample_datetime_docs, test_dataset_id):
    """Sample datetime dataset"""
    response = test_client.insert_documents(test_dataset_id, sample_datetime_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_numpy_dataset(test_client, sample_numpy_docs, test_dataset_id):
    """Sample numpy dataset"""
    response = test_client.insert_documents(test_dataset_id, sample_numpy_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_pandas_dataset(test_client, sample_pandas_docs, test_dataset_id):
    """Sample pandas dataset"""
    response = test_client.insert_documents(test_dataset_id, sample_pandas_docs)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)
