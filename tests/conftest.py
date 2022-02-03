"""All fixtures go here.
"""
import os
import pytest
import uuid
import random
import numpy as np
from relevanceai import Client
from relevanceai.http_client import Dataset
from datetime import datetime
import pandas as pd
import tempfile

from .utils import (
    generate_random_string,
    generate_random_vector,
    generate_random_label,
    generate_random_integer,
)

# We need to create separate datasets for different tests to avoid overwriting
# Our original database
# TODO: improve dataset modularisation
RANDOM_DATASET_SUFFIX = generate_random_string().lower()
RANDOM_PANDAS_DATASET_SUFFIX = generate_random_string().lower()
SAMPLE_DATASET_DATASET_PREFIX = "_sample_test_dataset_"
CLUSTER_DATASET_ID = SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()
LABEL_DATSET_ID = SAMPLE_DATASET_DATASET_PREFIX + generate_random_string().lower()


@pytest.fixture(scope="session", autouse=True)
def test_project():
    # test projects
    return os.getenv("TEST_PROJECT")


@pytest.fixture(scope="session", autouse=True)
def test_api_key():
    return os.getenv("TEST_API_KEY")


@pytest.fixture(scope="session", autouse=True)
def simple_doc():
    return [
        {
            "_id": uuid.uuid4().__str__(),
            "value": random.randint(0, 1000),
            "sample_1_vector_": generate_random_vector(N=100),
        }
    ]


@pytest.fixture(scope="session", autouse=True)
def test_client(test_project, test_api_key):
    client = Client(test_project, test_api_key)
    return client


@pytest.fixture(scope="session", autouse=True)
def test_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + RANDOM_DATASET_SUFFIX


@pytest.fixture(scope="session", autouse=True)
def pandas_test_dataset_id():
    return SAMPLE_DATASET_DATASET_PREFIX + RANDOM_PANDAS_DATASET_SUFFIX


@pytest.fixture(scope="session", autouse=True)
def test_sample_dataset(test_client: Client, simple_doc, test_dataset_id):
    """Sample dataset to insert and then delete"""
    simple_documents = simple_doc * 1000
    response = test_client._insert_documents(test_dataset_id, simple_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session", autouse=True)
def test_large_sample_dataset(test_client: Client, simple_doc, test_dataset_id):
    """Sample dataset to insert and then delete"""
    simple_documents = simple_doc * 1000
    response = test_client._insert_documents(test_dataset_id, simple_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session", autouse=True)
def error_doc():
    return [{"_id": 3, "value": np.nan}]


@pytest.fixture(scope="session", autouse=True)
def sample_vector_documents():
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
            "sample_1_value": generate_random_integer(N=100),
            "sample_2_value": generate_random_integer(N=100),
            "sample_3_value": generate_random_integer(N=100),
            "_chunk_": [
                {
                    "label": generate_random_label(),
                    "label_chunkvector_": generate_random_vector(100),
                }
            ],
        }

    N = 100
    return [_sample_vector_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session", autouse=True)
def sample_datetime_documents():
    def _sample_datetime_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_datetime": datetime.now(),
            "sample_2_datetime": datetime.now(),
        }

    N = 20
    return [_sample_datetime_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session", autouse=True)
def sample_numpy_documents():
    def _sample_numpy_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_numpy": np.random.randint(5, size=1)[0],
            "sample_2_numpy": np.random.rand(3, 2),
            "sample_3_numpy": np.nan,
        }

    N = 20
    return [_sample_numpy_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session", autouse=True)
def sample_pandas_documents():
    def _sample_numpy_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1_pandas": pd.DataFrame(
                np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
            ),
            "sample_2_pandas": pd.DataFrame(
                np.random.randint(0, 10, size=(10, 4)), columns=list("ABCD")
            ),
            "sample_3_pandas": pd.DataFrame(
                np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]),
                columns=["a", "b", "c"],
            ),
        }

    N = 20
    return [_sample_numpy_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)]


@pytest.fixture(scope="session", autouse=True)
def sample_nested_assorted_documents():
    def _sample_nested_assorted_doc(doc_id: str):
        return {
            "_id": doc_id,
            "sample_1": {
                "panda": pd.DataFrame(
                    np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
                ),
                "datetime": datetime.now(),
                "numpy": np.random.rand(3, 2),
            },
            "sample_2": [
                {
                    "panda": pd.DataFrame(
                        np.random.randint(0, 20, size=(20, 4)), columns=list("ABCD")
                    ),
                    "datetime": datetime.now(),
                    "numpy": np.random.rand(3, 2),
                },
                {
                    "panda": pd.DataFrame(
                        np.array([[1, 2, np.nan], [4, np.nan, 6], [7, 8, 9]]),
                        columns=["a", "b", "c"],
                    ),
                    "datetime": datetime.now(),
                    "numpy": np.random.rand(3, 2),
                },
            ],
        }

    N = 20
    return [
        _sample_nested_assorted_doc(doc_id=uuid.uuid4().__str__()) for _ in range(N)
    ]


@pytest.fixture(scope="session")
def test_sample_vector_dataset(
    test_client: Client, sample_vector_documents, test_dataset_id
):
    """
    Use this dataset if you just want vector
    """
    response = test_client._insert_documents(test_dataset_id, sample_vector_documents)
    yield test_dataset_id
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_clustered_dataset(test_client: Client, sample_vector_documents):
    """
    Use this test dataset if you want a dataset with clusters already.
    """
    test_client._insert_documents(CLUSTER_DATASET_ID, sample_vector_documents)
    test_client.vector_tools.cluster.kmeans_cluster(
        dataset_id=CLUSTER_DATASET_ID,
        vector_fields=["sample_1_vector_"],
        k=10,
        alias="kmeans_10",
        overwrite=True,
    )
    yield CLUSTER_DATASET_ID
    test_client.datasets.delete(CLUSTER_DATASET_ID)


@pytest.fixture(scope="session")
def test_datetime_dataset(
    test_client: Client, sample_datetime_documents, test_dataset_id
):
    """Sample datetime dataset"""
    response = test_client._insert_documents(test_dataset_id, sample_datetime_documents)
    yield response, len(sample_datetime_documents)
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_numpy_dataset(test_client: Client, sample_numpy_documents, test_dataset_id):
    """Sample numpy dataset"""
    response = test_client._insert_documents(test_dataset_id, sample_numpy_documents)
    yield response, len(sample_numpy_documents)
    test_client.datasets.delete(test_dataset_id)


@pytest.fixture(scope="session")
def test_pandas_dataset(
    test_client: Client, sample_pandas_documents, pandas_test_dataset_id
):
    """Sample pandas dataset"""
    response = test_client._insert_documents(
        pandas_test_dataset_id, sample_pandas_documents
    )
    yield response, len(sample_pandas_documents)
    test_client.datasets.delete(pandas_test_dataset_id)


@pytest.fixture(scope="session")
def test_nested_assorted_dataset(
    test_client: Client, sample_nested_assorted_documents, test_dataset_id
):
    """Sample nested assorted dataset"""
    response = test_client._insert_documents(
        test_dataset_id, sample_nested_assorted_documents
    )
    yield response, len(sample_nested_assorted_documents)
    test_client.datasets.delete(test_dataset_id)


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
def test_dataset_df(test_client: Client, test_sample_vector_dataset):
    df = test_client.Dataset(test_sample_vector_dataset)
    return df


@pytest.fixture(scope="session")
def test_csv_df(test_dataset_df: Dataset, sample_vector_documents, test_dataset_id):
    """Sample csv dataset"""
    with tempfile.NamedTemporaryFile(mode="w", delete=False) as csvfile:
        df = pd.DataFrame(sample_vector_documents)
        df.to_csv(csvfile)

        response = test_dataset_df.insert_csv(csvfile.name)
        yield response, len(sample_vector_documents)
        test_dataset_df.delete()
